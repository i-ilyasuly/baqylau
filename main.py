import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
import requests, io, datetime, os, time, threading
from fastapi import FastAPI
import uvicorn

# ── Баптау ──
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID", "0"))
RTSP_URL = os.environ.get("RTSP_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Firebase ──
cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Gemini ──
client = genai.Client(api_key=GEMINI_API_KEY)

# ── FastAPI — Cloud Run үшін міндетті ──
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Baqylau жұмыс жасап тұр! 🎥"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Telegram функциялары ──
def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text})

def send_photo(img_pil, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    buf.seek(0)
    requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                  files={"photo": buf})

def save_event(name, confidence):
    now = datetime.datetime.now()
    db.collection("events").add({
        "name": name, "event": "анықталды",
        "confidence": confidence,
        "timestamp": now,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    })

def load_known_faces():
    known = {}
    faces_ref = db.collection("known_faces").get()
    for doc in faces_ref:
        data = doc.to_dict()
        known[data["name"]] = np.array(data["encoding"])
    print(f"✅ {len(known)} адам жүктелді: {list(known.keys())}")
    return known

# ── Камера циклы — бөлек thread-та жұмыс жасайды ──
def camera_loop():
    print("📹 Камера циклы басталды!")
    known_faces = load_known_faces()
    
    if not RTSP_URL:
        print("⚠️ RTSP_URL орнатылмаған!")
        return

    cap = cv2.VideoCapture(RTSP_URL)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Камера ажырады, қайта қосылуда...")
            time.sleep(5)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        frame_count += 1
        if frame_count % int(fps) != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        if not face_locs:
            continue

        img_pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img_pil)
        detected = []

        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            name = "Белгісіз"
            best = 0.5
            for kname, kenc in known_faces.items():
                dist = face_recognition.face_distance([kenc], enc)[0]
                if dist < best:
                    best = dist
                    name = kname

            color = "green" if name != "Белгісіз" else "red"
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.rectangle([left, bottom, right, bottom+30], fill=color)
            draw.text((left+5, bottom+5), f"👤 {name}", fill="white")
            detected.append(name)
            save_event(name, round(1-best, 2))

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        caption = f"📹 Baqylau бақылауы\n⏱️ {now_str}\n\n"
        for n in detected:
            caption += f"✅ {n}!\n" if n != "Белгісіз" else f"⚠️ Белгісіз адам!\n"

        send_photo(img_pil, caption=caption)
        print(f"✅ {now_str}: {detected}")

# ── Іске қосу ──
if __name__ == "__main__":
    # Камера циклын бөлек thread-та іске қос
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    print("🚀 Baqylau сервері іске қосылды!")
    
    # Cloud Run PORT-ты оқиды
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
