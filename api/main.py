# ============================================================
# NUTRITRACK AI — FastAPI
# Cara jalankan: uvicorn main:app --reload
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ============================================================
# CUSTOM COMPONENTS (wajib didefinisikan ulang untuk load model)
# ============================================================
@tf.keras.utils.register_keras_serializable(package='Custom')
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(
            units=input_shape[-1], activation='softmax'
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        scores = self.attention_dense(inputs)
        return inputs * scores

    def get_config(self):
        return super(AttentionLayer, self).get_config()


@tf.keras.utils.register_keras_serializable(package='Custom')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=7)
        y_pred        = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor  = tf.pow(1.0 - y_pred, self.gamma)
        focal_loss    = self.alpha * focal_factor * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config


# ============================================================
# KONSTANTA
# ============================================================
TARGET_NAMES = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I',  'Overweight_Level_II',
    'Obesity_Type_I',      'Obesity_Type_II', 'Obesity_Type_III'
]

TARGET_LABELS = {
    'Insufficient_Weight': 'Berat Badan Kurang',
    'Normal_Weight':       'Berat Badan Normal',
    'Overweight_Level_I':  'Kelebihan Berat Badan Tingkat I',
    'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
    'Obesity_Type_I':      'Obesitas Tipe I',
    'Obesity_Type_II':     'Obesitas Tipe II',
    'Obesity_Type_III':    'Obesitas Tipe III',
}

# Saran statis sebagai fallback kalau Gemini gagal/limit
STATIC_ADVICE = {
    'Insufficient_Weight': """**Pola Makan**
• Tambah asupan kalori 300–500 kkal/hari dari sumber sehat
• Perbanyak protein: telur, ayam, ikan, kacang-kacangan
• Konsumsi karbohidrat kompleks: nasi merah, oat, ubi

**Aktivitas Fisik**
• Fokus ke latihan kekuatan (strength training) 3x seminggu
• Hindari kardio berlebihan yang membakar terlalu banyak kalori

**Gaya Hidup**
• Makan 5–6 kali sehari dalam porsi lebih kecil tapi sering
• Pantau berat badan setiap minggu

**Catatan Penting**
Konsultasikan dengan dokter gizi untuk program penambahan berat badan yang aman.""",

    'Normal_Weight': """**Pola Makan**
• Pertahankan pola makan seimbang dengan gizi lengkap
• Konsumsi sayur dan buah minimal 5 porsi per hari
• Minum air putih minimal 8 gelas per hari

**Aktivitas Fisik**
• Olahraga rutin 3–4x seminggu minimal 30 menit
• Kombinasikan kardio dan latihan kekuatan

**Gaya Hidup**
• Tetap aktif bergerak di sela-sela aktivitas harian
• Lakukan medical check-up rutin setahun sekali

**Catatan Penting**
Pertahankan gaya hidup sehatmu — kamu sudah di jalur yang benar!""",

    'Overweight_Level_I': """**Pola Makan**
• Kurangi asupan kalori 300–500 kkal/hari secara bertahap
• Hindari makanan tinggi gula: minuman manis, kue, permen
• Perbanyak sayuran dan protein tanpa lemak

**Aktivitas Fisik**
• Mulai dengan jalan kaki 30 menit setiap hari
• Olahraga minimal 4x seminggu

**Gaya Hidup**
• Hindari makan larut malam (setelah pukul 20.00)
• Gunakan piring lebih kecil untuk kontrol porsi

**Catatan Penting**
Ini saat yang tepat untuk mulai perubahan — perubahan kecil hari ini berdampak besar ke depannya!""",

    'Overweight_Level_II': """**Pola Makan**
• Kurangi asupan kalori 500–750 kkal/hari dengan panduan ahli gizi
• Hindari semua makanan ultra-processed dan fast food
• Catat asupan makanan harian (food journal)

**Aktivitas Fisik**
• Olahraga 5x seminggu minimal 45 menit per sesi
• Kombinasi kardio intensitas sedang dan latihan beban

**Gaya Hidup**
• Target penurunan berat badan 0.5–1 kg per minggu
• Cek kadar kolesterol dan gula darah secara rutin

**Catatan Penting**
Perubahan gaya hidup serius sangat dianjurkan — konsultasi dengan ahli gizi untuk panduan personal.""",

    'Obesity_Type_I': """**Pola Makan**
• Ikuti program diet terstruktur dengan panduan ahli gizi
• Hindari total: minuman manis, gorengan, fast food
• Makan dengan perlahan dan mindful eating

**Aktivitas Fisik**
• Olahraga minimal 5x seminggu, 60 menit per sesi
• Mulai dengan olahraga low-impact: renang, bersepeda

**Gaya Hidup**
• Tingkatkan aktivitas fisik harian: naik tangga, jalan kaki
• Evaluasi kesehatan mental — stres bisa memperburuk kondisi

**Catatan Penting**
WAJIB konsultasi dokter sebelum memulai program apapun.""",

    'Obesity_Type_II': """**Pola Makan**
• Diet harus dalam pengawasan ketat dokter dan ahli gizi
• Fokus pada makanan anti-inflamasi: ikan, sayur hijau, berry
• Hindari total alkohol dan minuman manis

**Aktivitas Fisik**
• Olahraga HANYA dengan supervisi profesional
• Renang sangat dianjurkan karena minim tekanan pada sendi

**Gaya Hidup**
• Evaluasi komplikasi: sleep apnea, diabetes, hipertensi
• Dukungan psikologis sangat dianjurkan

**Catatan Penting**
SEGERA konsultasi dokter spesialis gizi atau endokrinologi.""",

    'Obesity_Type_III': """**Pola Makan**
• Diet HARUS di bawah pengawasan ketat tim medis
• Suplemen vitamin dan mineral wajib karena risiko defisiensi
• Tidak dianjurkan melakukan diet sendiri tanpa panduan dokter

**Aktivitas Fisik**
• Aktivitas fisik HANYA atas rekomendasi dokter
• Hidroterapi bisa menjadi pilihan aman

**Gaya Hidup**
• Libatkan keluarga dalam proses pemulihan
• Dukungan psikiatri/psikologi sangat penting

**Catatan Penting**
SEGERA temui dokter spesialis — kondisi ini memerlukan penanganan medis segera."""
}

# Path model
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'nutritrack_model_gradienttape.keras')


# ============================================================
# LOAD MODEL
# ============================================================
try:
    model = load_model(
        MODEL_PATH,
        custom_objects={
            'AttentionLayer': AttentionLayer,
            'FocalLoss'     : FocalLoss
        }
    )
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    model = None


# ============================================================
# SETUP GEMINI
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API siap.")
    except Exception as e:
        gemini_client = None
        print(f"⚠️ Gemini gagal diinisialisasi: {e}")
else:
    gemini_client = None
    print("⚠️ GEMINI_API_KEY tidak ditemukan di .env")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title       = "NutriTrack AI API",
    description = "API prediksi risiko obesitas berbasis Deep Learning + saran Generative AI",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ============================================================
# SCHEMA INPUT
# ============================================================
class PredictRequest(BaseModel):
    gender         : int   = Field(..., ge=0, le=1,   description="0=Female, 1=Male")
    age            : float = Field(..., ge=14, le=61,  description="Usia (14-61 tahun)")
    height         : float = Field(..., ge=145, le=198, description="Tinggi badan dalam cm")
    weight         : float = Field(..., ge=39, le=173,  description="Berat badan dalam kg")
    family_history : int   = Field(..., ge=0, le=1,   description="Riwayat keluarga obesitas: 0=Tidak, 1=Ya")
    high_cal_food  : int   = Field(..., ge=0, le=1,   description="Konsumsi makanan tinggi kalori: 0=Tidak, 1=Ya")
    veg_freq       : float = Field(..., ge=1, le=5,   description="Frekuensi konsumsi sayur (1-5)")
    meals_day      : float = Field(..., ge=1, le=6,   description="Jumlah makan per hari (1-6)")
    snack          : int   = Field(..., ge=0, le=3,   description="Frekuensi ngemil: 0=Tidak, 1=Kadang, 2=Sering, 3=Selalu")
    smoking        : int   = Field(..., ge=0, le=1,   description="Merokok: 0=Tidak, 1=Ya")
    water          : float = Field(..., ge=1, le=5,   description="Konsumsi air harian (1-5)")
    cal_monitoring : int   = Field(..., ge=0, le=1,   description="Monitoring kalori: 0=Tidak, 1=Ya")
    faf            : int   = Field(..., ge=0, le=7,   description="Frekuensi olahraga per minggu (hari)")
    tue            : int   = Field(..., ge=0, le=12,  description="Waktu pakai teknologi per hari (jam)")
    alcohol        : int   = Field(..., ge=0, le=3,   description="Konsumsi alkohol: 0=Tidak, 1=Kadang, 2=Sering, 3=Selalu")
    transport      : int   = Field(..., ge=0, le=4,   description="Transportasi: 0=Mobil, 1=Sepeda, 2=Motor, 3=Umum, 4=Jalan Kaki")


# ============================================================
# HELPER — PREPROCESSING
# ============================================================
def scale(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def preprocess(req: PredictRequest) -> np.ndarray:
    height_m   = req.height / 100
    input_data = [
        req.gender,
        scale(req.age,      14, 61),
        scale(height_m,     1.45, 1.98),
        scale(req.weight,   39, 173),
        req.family_history,
        req.high_cal_food,
        scale(req.veg_freq,  1, 5),
        scale(req.meals_day, 1, 6),
        req.snack,
        req.smoking,
        scale(req.water, 1, 5),
        req.cal_monitoring,
        scale(req.faf, 0, 7),
        scale(req.tue, 0, 12),
        req.alcohol,
        req.transport,
    ]
    return np.array(input_data, dtype=np.float32).reshape(1, -1)


# ============================================================
# HELPER — GEMINI SARAN (dengan fallback ke saran statis)
# ============================================================
def get_advice(label: str, confidence: float, req: PredictRequest) -> dict:
    bmi = req.weight / ((req.height / 100) ** 2)

    # Coba Gemini dulu
    if gemini_client is not None:
        prompt = f"""
Kamu adalah asisten kesehatan AI bernama NutriTrack. Berikan saran gaya hidup yang personal, 
empatik, dan actionable dalam Bahasa Indonesia untuk pengguna berikut:

Data Pengguna:
- Hasil Klasifikasi: {TARGET_LABELS.get(label, label)} (confidence: {confidence*100:.1f}%)
- BMI: {bmi:.1f}
- Usia: {req.age} tahun
- Jenis Kelamin: {"Laki-laki" if req.gender == 1 else "Perempuan"}
- Riwayat keluarga obesitas: {"Ya" if req.family_history == 1 else "Tidak"}
- Frekuensi olahraga: {req.faf} hari/minggu
- Konsumsi air: {req.water}/5
- Monitoring kalori: {"Ya" if req.cal_monitoring == 1 else "Tidak"}
- Merokok: {"Ya" if req.smoking == 1 else "Tidak"}

Berikan saran dalam format berikut (singkat, maksimal 3 poin per bagian):
**Pola Makan** (2-3 saran spesifik)
**Aktivitas Fisik** (2-3 saran spesifik)
**Gaya Hidup** (2-3 saran spesifik)
**Catatan Penting** (1 kalimat motivasi)

Jangan ulangi hasil klasifikasi. Fokus pada saran praktis yang bisa langsung dilakukan.
"""
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return {
                "advice" : response.text,
                "source" : "gemini"
            }
        except Exception:
            pass  # Lanjut ke fallback

    # Fallback ke saran statis
    return {
        "advice" : STATIC_ADVICE.get(label, "Konsultasikan kondisi Anda dengan dokter atau ahli gizi."),
        "source" : "static"
    }


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "app"    : "NutriTrack AI API",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs"
    }

@app.get("/health")
def health():
    return {
        "model_loaded" : model is not None,
        "gemini_ready" : gemini_client is not None,
        "model_path"   : MODEL_PATH,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Cek path model.")

    # Preprocessing & prediksi
    input_array = preprocess(req)
    proba       = model.predict(input_array, verbose=0)[0]
    pred_idx    = int(np.argmax(proba))
    pred_name   = TARGET_NAMES[pred_idx]
    confidence  = float(proba[pred_idx])

    # Semua probabilitas
    all_proba = {
        TARGET_NAMES[i]: round(float(proba[i]), 4)
        for i in range(len(TARGET_NAMES))
    }

    # Saran (Gemini atau fallback statis)
    advice_result = get_advice(pred_name, confidence, req)

    return {
        "prediction"   : pred_name,
        "label"        : TARGET_LABELS.get(pred_name, pred_name),
        "confidence"   : round(confidence, 4),
        "probabilities": all_proba,
        "bmi"          : round(req.weight / ((req.height / 100) ** 2), 2),
        "ai_advice"    : advice_result["advice"],
        "advice_source": advice_result["source"],  # "gemini" atau "static"
    }
