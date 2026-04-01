import os, json, time, threading, datetime, requests
import cv2, numpy as np, face_recognition, firebase_admin

from firebase_admin import credentials, firestore
from google.cloud import secretmanager
from ultralytics import YOLO
from google import genai
from google.genai import types
from fastapi import FastAPI, Request  # ← Request қосылды (webhook үшін)
import uvicorn

BOT_TOKEN  = os.environ.get("BOT_TOKEN", "")
CHAT_ID    = os.environ.get("CHAT_ID", "")
RTSP_URL   = os.environ.get("RTSP_URL", "test")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
PROJECT_ID = os.environ.get("GCP_PROJECT", "baqylau-491808")

# ── ГЛОБАЛ АЙНЫМАЛЫЛАР ─────────────────────────────────────
db            = None
yolo_model    = None
gemini_client = None
known_face_encodings = []
known_face_names     = []
app_ready     = False

# ── FASTAPI — БІРДЕН ІСКЕ ҚОСЫЛАДЫ ────────────────────────
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Baqylau жұмыс жасап тұр 🚀", "ready": app_ready}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── TELEGRAM WEBHOOK — СҰРАҚ ҚАБЫЛДАЙДЫ ──────────────────
@app.post("/webhook")
async def webhook(request: Request):
    # Telegram-дан келген JSON деректерді оқу
    data = await request.json()

    # Хабар мен chat_id алу
    msg     = data.get("message", {})
    text    = msg.get("text", "").lower()  # ← кіші әріпке — сұрақты оңай табу үшін
    chat_id = msg.get("chat", {}).get("id")

    if not chat_id or not text:
        return {"ok": True}  # ← бос хабар болса өткіз

    # ─── СҰРАҚ-ЖАУАП ЛОГИКАСЫ ────────────────────────────
    if any(w in text for w in ["кім келді", "кім бар", "кто пришел"]):
        # Бүгінгі оқиғаларды Firestore-дан оқып жауап береді
        answer = get_today_events()
        send_message_to(chat_id, answer)

    elif any(w in text for w in ["сәлем", "сәлеметсіз", "привет", "/start"]):
        send_message_to(chat_id,
            "Сәлем! 👋 Мен Baqylau — үйіңнің ақылды қарауылымын!\n\n"
            "Сұрай аласың:\n"
            "• Бүгін кім келді?\n"
            "• Апа нешеде келді?\n"
            "• Белгісіз адам болды ма?"
        )

    else:
        # Басқа сұрақтарды Gemini арқылы жауаптайды
        events = get_today_events()
        prompt = f"Бүгінгі оқиғалар: {events}\n\nСұрақ: {text}\n\nҚазақ тілінде қысқа жауап бер."
        answer = describe_with_gemini_text(prompt)
        send_message_to(chat_id, answer)

    return {"ok": True}

# ── БАРЛЫҚ АУЫР ЖҮКТЕМЕ — ФОНДА ───────────────────────────
def initialize():
    global db, yolo_model, gemini_client, app_ready

    print("🔄 Инициализация басталды...")

    # Secret Manager → Firestore кілтін қауіпсіз алу
    try:
        sm_client   = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{PROJECT_ID}/secrets/firestore-sa-key/versions/latest"
        response    = sm_client.access_secret_version(request={"name": secret_name})
        sa_info     = json.loads(response.payload.data.decode("UTF-8"))
        cred        = credentials.Certificate(sa_info)
        firebase_admin.initialize_app(cred)
        db          = firestore.client()
        print("✅ Firestore қосылды!")
    except Exception as e:
        print(f"❌ Firestore қате: {e}")

    # YOLO — адам, объект табу моделі
    try:
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO жүктелді!")
    except Exception as e:
        print(f"❌ YOLO қате: {e}")

    # Gemini — AI сипаттама беру клиенті
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini қосылды!")
    except Exception as e:
        print(f"❌ Gemini қате: {e}")

    # known_faces — таныс адамдар тізімін Firestore-дан жүктеу
    if db:
        load_known_faces()

    app_ready = True
    print("🚀 Baqylau толық дайын!")

    # Камера циклы — фонда үздіксіз жұмыс жасайды
    camera_loop()


def load_known_faces():
    # Firestore-дан Апа, Сезім т.б. бет деректерін жүктеу
    global known_face_encodings, known_face_names
    try:
        docs = db.collection("known_faces").stream()
        for doc in docs:
            data = doc.to_dict()
            known_face_encodings.append(np.array(data["encoding"]))
            known_face_names.append(data["name"])
        print(f"✅ {len(known_face_names)} адам жүктелді: {known_face_names}")
    except Exception as e:
        print(f"❌ known_faces қате: {e}")


# ── TELEGRAM ФУНКЦИЯЛАРЫ ───────────────────────────────────

def send_message(text):
    # Негізгі CHAT_ID-ге жіберу (камера хабарлары үшін)
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text})
    except Exception as e:
        print(f"❌ Telegram қате: {e}")


def send_message_to(chat_id, text):
    # Webhook сұрақтарына жауап беру — кез-келген chat_id-ге
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": chat_id, "text": text})
    except Exception as e:
        print(f"❌ Telegram қате: {e}")


def send_photo(img_bytes, caption=""):
    # Негізгі CHAT_ID-ге фото жіберу (камерадан сурет)
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                      data={"chat_id": CHAT_ID, "caption": caption},
                      files={"photo": ("frame.jpg", img_bytes, "image/jpeg")})
    except Exception as e:
        print(f"❌ Telegram фото қате: {e}")


# ── FIRESTORE ФУНКЦИЯЛАРЫ ──────────────────────────────────

def save_event(name):
    # Адам танылғанда оқиғаны Firestore-ға жазу
    if not db:
        return
    try:
        now = datetime.datetime.now()
        db.collection("events").add({
            "name": name,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": now
        })
    except Exception as e:
        print(f"❌ Firestore жазу қате: {e}")


def get_today_events():
    # Бүгінгі оқиғаларды Firestore-дан оқу
    if not db:
        return "Деректер қорына қосылу мүмкін болмады"
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        docs  = db.collection("events").stream()

        # Python-да сүзу — composite index жасамай-ақ жұмыс жасайды
        today_events = []
        for doc in docs:
            d = doc.to_dict()
            if d.get("date") == today:
                today_events.append(f"👤 {d['name']} — {d['time']}-де")

        if today_events:
            return "📋 Бүгін үйде болғандар:\n" + "\n".join(today_events)
        else:
            return "Бүгін ешкім тіркелмеді"
    except Exception as e:
        return f"Қате: {e}"


def is_recently_seen(name):
    # Firestore-дан соңғы көру уақытын оқу
    # Егер 300 секунд (5 мин) өтпесе — True қайтарады, хабар жібермейді
    # 25-ҚАДАМ: бұрын last_seen={} тек жадта болатын, Cloud Run қайта іске қосылса тазаланатын
    # Енді Firestore-да сақталады — Cloud Run рестарт болса да жұмыс жасайды
    if not db:
        return False
    try:
        doc = db.collection("last_seen").document(name).get()
        if doc.exists:
            last_ts = doc.to_dict().get("timestamp", 0)
            if time.time() - last_ts < 300:
                return True  # ← жақында көрілген, қайта хабарлама жібермейді
    except:
        pass
    return False  # ← жаңа немесе 5 минут өткен, хабарлама жіберуге болады


def set_last_seen(name):
    # Адам танылған соң Firestore-ға соңғы уақытын жазу
    if not db:
        return
    try:
        db.collection("last_seen").document(name).set({
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"❌ last_seen жазу қате: {e}")


# ── GEMINI ФУНКЦИЯЛАРЫ ─────────────────────────────────────

def describe_with_gemini(frame):
    # Камера кадрын суретпен Gemini-ге жіберіп қазақша сипаттама алу
    if not gemini_client:
        return ""
    try:
        _, buf = cv2.imencode(".jpg", frame)
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                "Осы кадрда не болып жатыр? Қазақ тілінде қысқаша сипатта."
            ]
        )
        return resp.text
    except Exception as e:
        return f"Gemini қате: {e}"


def describe_with_gemini_text(prompt):
    # Мәтін сұрақ қою — webhook жауаптары үшін
    if not gemini_client:
        return "Gemini қосылмаған"
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        return resp.text
    except Exception as e:
        return f"Gemini қате: {e}"


# ── КАМЕРА ЦИКЛЫ ───────────────────────────────────────────

def camera_loop():
    if RTSP_URL == "test":
        # Нақты камера жоқ кезде — күту режимі
        print("⚠️ RTSP_URL='test' — нақты камера жоқ, күту режимінде...")
        while True:
            time.sleep(60)

    while True:
        try:
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                print("⚠️ Камера ажырады, қайта қосылуда...")
                time.sleep(5)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO — алдымен адам бар-жоғын тексеру (Gemini-ді үнемдеу)
                if yolo_model:
                    results = yolo_model(frame, verbose=False)
                    persons = [b for b in results[0].boxes
                               if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5]
                    if not persons:
                        time.sleep(1)
                        continue  # ← адам жоқ — келесі кадрға өт

                # Face Recognition — адамды атымен тану
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encs = face_recognition.face_encodings(rgb,
                                face_recognition.face_locations(rgb))

                for enc in face_encs:
                    name = "Белгісіз"

                    # Firestore-дан жүктелген таныс адамдармен салыстыру
                    if known_face_encodings:
                        dists = face_recognition.face_distance(known_face_encodings, enc)
                        best  = int(np.argmin(dists))
                        if dists[best] < 0.5:  # ← 0.5 шек мән — жақынырақ = дұрысырақ
                            name = known_face_names[best]

                    # 25-ҚАДАМ: is_recently_seen() — Firestore-дан тексереді
                    # Бұрын: last_seen[name] — тек жадта
                    # Енді: Firestore-да — Cloud Run рестарт болса да жұмыс жасайды
                    if is_recently_seen(name):
                        continue  # ← 5 минут өтпесе хабарлама жібермейді

                    # Соңғы уақытты Firestore-ға жазу
                    set_last_seen(name)

                    # Оқиғаны Firestore-ға жазу
                    save_event(name)

                    # Gemini — кадрды қазақша сипаттау
                    desc   = describe_with_gemini(frame)
                    t_str  = datetime.datetime.now().strftime("%H:%M")
                    _, buf = cv2.imencode(".jpg", frame)

                    # Telegram-ға фото + сипаттама жіберу
                    send_photo(buf.tobytes(), f"👤 {name} үйде! ({t_str})\n\n🤖 {desc}")

                time.sleep(1)
            cap.release()

        except Exception as e:
            print(f"❌ Камера циклы қате: {e}")
        time.sleep(5)


if __name__ == "__main__":
    # Фонда инициализация — FastAPI бірден іске қосылады (Cloud Run timeout болмасын)
    threading.Thread(target=initialize, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
