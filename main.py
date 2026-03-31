import os, json, time, threading, datetime, requests
import cv2, numpy as np, face_recognition, firebase_admin

from firebase_admin import credentials, firestore
from google.cloud import secretmanager
from ultralytics import YOLO
from google import genai
from google.genai import types
from fastapi import FastAPI
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

# ── БАРЛЫҚ АУЫР ЖҮКТЕМЕ — ФОНДА ───────────────────────────
def initialize():
    global db, yolo_model, gemini_client, app_ready

    print("🔄 Инициализация басталды...")

    # Secret Manager → Firestore
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

    # YOLO
    try:
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO жүктелді!")
    except Exception as e:
        print(f"❌ YOLO қате: {e}")

    # Gemini
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini қосылды!")
    except Exception as e:
        print(f"❌ Gemini қате: {e}")

    # known_faces жүктеу
    if db:
        load_known_faces()

    app_ready = True
    print("🚀 Baqylau толық дайын!")

    # Камера циклы
    camera_loop()

def load_known_faces():
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

def send_message(text):
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text})
    except Exception as e:
        print(f"❌ Telegram қате: {e}")

def send_photo(img_bytes, caption=""):
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                      data={"chat_id": CHAT_ID, "caption": caption},
                      files={"photo": ("frame.jpg", img_bytes, "image/jpeg")})
    except Exception as e:
        print(f"❌ Telegram фото қате: {e}")

def save_event(name):
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

def describe_with_gemini(frame):
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

last_seen = {}

def camera_loop():
    if RTSP_URL == "test":
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
                if yolo_model:
                    results = yolo_model(frame, verbose=False)
                    persons = [b for b in results[0].boxes
                               if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5]
                    if not persons:
                        time.sleep(1)
                        continue
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encs = face_recognition.face_encodings(rgb,
                                face_recognition.face_locations(rgb))
                for enc in face_encs:
                    name = "Белгісіз"
                    if known_face_encodings:
                        dists = face_recognition.face_distance(known_face_encodings, enc)
                        best  = int(np.argmin(dists))
                        if dists[best] < 0.5:
                            name = known_face_names[best]
                    now_ts = time.time()
                    if name in last_seen and now_ts - last_seen[name] < 300:
                        continue
                    last_seen[name] = now_ts
                    save_event(name)
                    desc   = describe_with_gemini(frame)
                    t_str  = datetime.datetime.now().strftime("%H:%M")
                    _, buf = cv2.imencode(".jpg", frame)
                    send_photo(buf.tobytes(), f"👤 {name} үйде! ({t_str})\n\n🤖 {desc}")
                time.sleep(1)
            cap.release()
        except Exception as e:
            print(f"❌ Камера циклы қате: {e}")
        time.sleep(5)

if __name__ == "__main__":
    # Фонда инициализация — FastAPI бірден іске қосылады
    threading.Thread(target=initialize, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
