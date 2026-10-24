import cv2
import time
import sqlite3
import os
import requests
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# ==========================================
# KONFIGURASI & VARIABEL GLOBAL (STATE)
# ==========================================
DB_FILE = "database.db"
MODEL_PATH = "best_web.pt" # Nama file model yang disimpan di server
ESP32_IP = "http://192.168.4.1" # Ganti dengan IP ESP32 milikmu nanti

yolo_model = None
state = "IDLE" # Status: IDLE, SCANNING, PAYING
cart = []
total_price = 0
last_item_add_time = 0
cooldown_time = 3.0 # Jeda 3 detik sebelum bisa scan barang lagi (menghindari spam)

# ==========================================
# DATABASE SETUP (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            price INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def update_db_from_model():
    """Mengekstrak nama label dari YOLO dan menyimpannya ke database"""
    global yolo_model
    if yolo_model is None: return
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # yolo_model.names berisi dictionary {0: 'produk_a', 1: 'produk_b'}
    for cls_id, name in yolo_model.names.items():
        try:
            # INSERT OR IGNORE memastikan tidak duplikat jika nama sudah ada
            c.execute("INSERT OR IGNORE INTO products (name, price) VALUES (?, ?)", (name, 0))
        except sqlite3.Error as e:
            print("DB Error:", e)
    conn.commit()
    conn.close()
    print("[INFO] Database produk berhasil diperbarui dari model AI.")

def get_product_price(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT price FROM products WHERE name = ?", (name,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

# Load model saat server pertama kali menyala (jika file .pt sudah ada)
if os.path.exists(MODEL_PATH):
    try:
        yolo_model = YOLO(MODEL_PATH)
        init_db()
        update_db_from_model()
    except Exception as e:
        print(f"[ERROR] Gagal memuat {MODEL_PATH}: {e}")
else:
    init_db()

# ==========================================
# KONTROL ESP32
# ==========================================
def send_esp32_cmd(endpoint):
    """Fungsi untuk mengirim trigger ke ESP32 tanpa memblokir sistem"""
    try:
        url = f"{ESP32_IP}/{endpoint}"
        # requests.get(url, timeout=1) # Dicomment agar tidak error saat testing tanpa ESP
        print(f"[ESP32 TRIGGER] Berhasil mengirim perintah: {url}")
    except Exception as e:
        print(f"[ESP32 ERROR] Gagal menghubungi ESP32: {e}")

# ==========================================
# LOGIKA WEBCAM & AI INFERENCE
# ==========================================
def generate_frames():
    global state, last_item_add_time, cart, total_price, yolo_model
    cap = cv2.VideoCapture(1) # Ganti 1 atau 2 jika pakai USB Webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Hanya jalankan AI jika sedang IDLE atau SCANNING
        if state in ["IDLE", "SCANNING"] and yolo_model is not None:
            results = yolo_model(frame, stream=True)
            
            for r in results:
                # PASTIKAN HANYA 1 PRODUK PER FRAME (Ambil yang confidence-nya paling tinggi)
                if len(r.boxes) > 0:
                    best_box = max(r.boxes, key=lambda b: b.conf[0])
                    conf = float(best_box.conf[0])
                    
                    if conf > 0.5: # Threshold keyakinan 50%
                        cls_id = int(best_box.cls[0])
                        name = yolo_model.names[cls_id]
                        
                        # Gambar kotak di layar
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Logika memasukkan ke keranjang
                        current_time = time.time()
                        if current_time - last_item_add_time > cooldown_time:
                            price = get_product_price(name)
                            cart.append({"name": name, "price": price})
                            total_price += price
                            last_item_add_time = current_time
                            
                            print(f"[SCAN] {name} ditambahkan. Total: Rp{total_price}")
                            
                            # Jika ini barang pertama, ubah state dan nyalakan alarm
                            if state == "IDLE":
                                state = "SCANNING"
                                send_esp32_cmd("en-alarm")

        # Cek hitung mundur 8 detik jika sedang scanning
        if state == "SCANNING":
            if time.time() - last_item_add_time > 8.0:
                print("[INFO] Waktu scan habis. Melanjutkan ke pembayaran.")
                state = "PAYING"

        # Encode gambar ke format JPEG untuk disiarkan ke Web HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# API ROUTING & ENDPOINTS
# ==========================================
@app.get("/")
def serve_frontend():
    # Menyajikan file index.html sebagai antarmuka utama
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/video_feed")
def video_feed():
    # Menyiarkan video secara real-time ke tag <img> di web
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/state")
def get_state():
    # API ini dipanggil HTML setiap 0.5 detik untuk sinkronisasi tampilan
    countdown = 0
    if state == "SCANNING":
        countdown = max(0, 8.0 - (time.time() - last_item_add_time))
        
    return JSONResponse({
        "state": state,
        "cart": cart,
        "total": total_price,
        "countdown": round(countdown, 1)
    })

@app.post("/api/admin/upload_model")
def upload_model(file: UploadFile = File(...)):
    global yolo_model
    # Simpan file yang diupload dan timpa yang lama
    with open(MODEL_PATH, "wb") as buffer:
        buffer.write(file.file.read())
    
    # Reload model YOLO secara dinamis (Hot-Reload)
    yolo_model = YOLO(MODEL_PATH)
    update_db_from_model() # Update database nama produk
    return JSONResponse({"status": "success", "message": "Model berhasil diunggah & DB di-update!"})

@app.get("/api/admin/products")
def get_products():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, price FROM products")
    products = [{"id": row[0], "name": row[1], "price": row[2]} for row in c.fetchall()]
    conn.close()
    return JSONResponse(products)

@app.post("/api/admin/update_price")
async def update_price(request: Request):
    # Digunakan untuk mengubah harga dari halaman admin
    data = await request.json()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE products SET price = ? WHERE id = ?", (data['price'], data['id']))
    conn.commit()
    conn.close()
    return JSONResponse({"status": "success"})

@app.post("/api/sim_payment")
def simulate_payment():
    """Endpoint untuk menyimulasikan pembayaran berhasil (midtrans webhook)"""
    global state, cart, total_price
    state = "IDLE"
    cart = []
    total_price = 0
    send_esp32_cmd("dis-alarm") # Matikan alarm
    return JSONResponse({"status": "success", "message": "Pembayaran Berhasil. Alarm dimatikan."})

# Cara Menjalankan: 
# pip install fastapi uvicorn python-multipart
# uvicorn app:app --host 0.0.0.0 --port 8000