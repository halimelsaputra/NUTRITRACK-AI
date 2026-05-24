# ============================================================
# NUTRITRACK AI — Dashboard Streamlit
# Cara jalankan: streamlit run nutritrack_dashboard.py
# ============================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title = "NutriTrack AI",
    page_icon  = "🥗",
    layout     = "wide"
)

# ============================================================
# CUSTOM COMPONENTS (harus didefinisikan ulang untuk load model)
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
        y_pred  = tf.cast(y_pred, tf.float32)
        y_true  = tf.cast(y_true, tf.float32)
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
# DATABASE REKOMENDASI
# ============================================================
REKOMENDASI = {
    'Insufficient_Weight': {
        'status'   : '⚠️ Berat Badan Kurang',
        'warna'    : '#FFA500',
        'deskripsi': 'Berat badanmu berada di bawah batas normal. Tubuh membutuhkan lebih banyak nutrisi.',
        'diet'     : [
            'Tambah asupan kalori 300–500 kkal/hari dari sumber sehat',
            'Perbanyak konsumsi protein: telur, ayam, ikan, kacang-kacangan',
            'Konsumsi karbohidrat kompleks: nasi merah, oat, ubi',
            'Makan 5–6 kali sehari dalam porsi lebih kecil tapi sering',
            'Tambahkan lemak sehat: alpukat, kacang almond, minyak zaitun',
        ],
        'olahraga' : [
            'Fokus ke latihan kekuatan (strength training) 3x seminggu',
            'Hindari kardio berlebihan yang membakar terlalu banyak kalori',
            'Yoga atau pilates untuk meningkatkan massa otot ringan',
        ],
        'medis'    : [
            'Konsultasi ke dokter gizi untuk program penambahan berat badan',
            'Cek kemungkinan gangguan tiroid atau masalah pencernaan',
            'Pertimbangkan suplemen jika direkomendasikan dokter',
        ]
    },
    'Normal_Weight': {
        'status'   : '✅ Berat Badan Normal',
        'warna'    : '#28a745',
        'deskripsi': 'Selamat! Berat badanmu berada di rentang ideal. Pertahankan gaya hidup sehatmu.',
        'diet'     : [
            'Pertahankan pola makan seimbang dengan gizi lengkap',
            'Konsumsi sayur dan buah minimal 5 porsi per hari',
            'Batasi makanan tinggi gula dan lemak jenuh',
            'Minum air putih minimal 8 gelas per hari',
            'Jangan skip sarapan untuk menjaga metabolisme',
        ],
        'olahraga' : [
            'Olahraga rutin 3–4x seminggu minimal 30 menit',
            'Kombinasikan kardio dan latihan kekuatan',
            'Tetap aktif bergerak di sela-sela aktivitas harian',
        ],
        'medis'    : [
            'Lakukan medical check-up rutin setahun sekali',
            'Pantau berat badan secara berkala setiap bulan',
        ]
    },
    'Overweight_Level_I': {
        'status'   : '⚠️ Kelebihan Berat Badan Tingkat I',
        'warna'    : '#FFC107',
        'deskripsi': 'Berat badanmu sedikit di atas normal. Ini saat yang tepat untuk mulai perubahan gaya hidup.',
        'diet'     : [
            'Kurangi asupan kalori 300–500 kkal/hari secara bertahap',
            'Hindari makanan tinggi gula: minuman manis, kue, permen',
            'Perbanyak sayuran dan protein tanpa lemak',
            'Hindari makan larut malam (setelah pukul 20.00)',
            'Gunakan piring lebih kecil untuk kontrol porsi',
        ],
        'olahraga' : [
            'Mulai dengan jalan kaki 30 menit setiap hari',
            'Tingkatkan intensitas secara bertahap ke jogging ringan',
            'Olahraga minimal 4x seminggu',
        ],
        'medis'    : [
            'Konsultasi ke dokter atau ahli gizi untuk program diet',
            'Pantau tekanan darah dan kadar gula darah secara rutin',
        ]
    },
    'Overweight_Level_II': {
        'status'   : '⚠️ Kelebihan Berat Badan Tingkat II',
        'warna'    : '#FF8C00',
        'deskripsi': 'Berat badanmu cukup jauh di atas normal. Perubahan gaya hidup serius sangat dianjurkan.',
        'diet'     : [
            'Kurangi asupan kalori 500–750 kkal/hari dengan panduan ahli gizi',
            'Hindari semua makanan ultra-processed dan fast food',
            'Konsumsi makanan tinggi serat: sayur, buah, biji-bijian',
            'Catat asupan makanan harian (food journal)',
        ],
        'olahraga' : [
            'Olahraga 5x seminggu minimal 45 menit per sesi',
            'Kombinasi kardio intensitas sedang dan latihan beban',
            'Target penurunan berat badan 0.5–1 kg per minggu',
        ],
        'medis'    : [
            'Wajib konsultasi ke dokter dan ahli gizi',
            'Cek kadar kolesterol, gula darah, dan tekanan darah',
        ]
    },
    'Obesity_Type_I': {
        'status'   : '🔴 Obesitas Tipe I',
        'warna'    : '#DC3545',
        'deskripsi': 'Kamu mengalami obesitas tingkat pertama. Penanganan segera sangat dianjurkan.',
        'diet'     : [
            'Ikuti program diet terstruktur dengan panduan ahli gizi',
            'Hindari total: minuman manis, gorengan, fast food',
            'Perbanyak protein untuk menjaga massa otot saat diet',
            'Makan dengan perlahan dan mindful eating',
        ],
        'olahraga' : [
            'Olahraga minimal 5x seminggu, 60 menit per sesi',
            'Mulai dengan olahraga low-impact: renang, bersepeda',
            'Tingkatkan aktivitas fisik harian: naik tangga, jalan kaki',
        ],
        'medis'    : [
            'WAJIB konsultasi dokter sebelum memulai program apapun',
            'Periksa risiko diabetes tipe 2, hipertensi, dan penyakit jantung',
            'Evaluasi kesehatan mental — stres bisa memperburuk obesitas',
        ]
    },
    'Obesity_Type_II': {
        'status'   : '🔴 Obesitas Tipe II',
        'warna'    : '#B22222',
        'deskripsi': 'Kamu mengalami obesitas tingkat dua. Kondisi ini membutuhkan penanganan medis serius.',
        'diet'     : [
            'Diet harus dalam pengawasan ketat dokter dan ahli gizi',
            'Fokus pada makanan anti-inflamasi: ikan, sayur hijau, berry',
            'Hindari total alkohol dan minuman manis',
        ],
        'olahraga' : [
            'Olahraga HANYA dengan supervisi profesional',
            'Mulai sangat perlahan: jalan kaki 15–20 menit',
            'Renang sangat dianjurkan karena minim tekanan pada sendi',
        ],
        'medis'    : [
            'SEGERA konsultasi dokter spesialis gizi atau endokrinologi',
            'Evaluasi komplikasi: sleep apnea, diabetes, hipertensi',
            'Diskusikan kemungkinan terapi medis atau bedah bariatrik',
            'Dukungan psikologis sangat dianjurkan',
        ]
    },
    'Obesity_Type_III': {
        'status'   : '🚨 Obesitas Tipe III (Obesitas Morbid)',
        'warna'    : '#8B0000',
        'deskripsi': 'Ini adalah tingkat obesitas paling berat. Intervensi medis segera sangat diperlukan.',
        'diet'     : [
            'Diet HARUS di bawah pengawasan ketat tim medis',
            'Suplemen vitamin dan mineral wajib karena risiko defisiensi',
            'Tidak dianjurkan melakukan diet sendiri tanpa panduan dokter',
        ],
        'olahraga' : [
            'Aktivitas fisik HANYA atas rekomendasi dokter',
            'Terapi fisik/fisioterapi sebagai langkah awal',
            'Hidroterapi bisa menjadi pilihan aman',
        ],
        'medis'    : [
            'SEGERA temui dokter spesialis — ini kondisi darurat kesehatan',
            'Evaluasi menyeluruh: jantung, paru, diabetes, kolesterol',
            'Operasi bariatrik perlu didiskusikan dengan dokter',
            'Dukungan psikiatri/psikologi sangat penting',
            'Libatkan keluarga dalam proses pemulihan',
        ]
    }
}

TARGET_NAMES = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I',  'Overweight_Level_II',
    'Obesity_Type_I',      'Obesity_Type_II', 'Obesity_Type_III'
]

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_nutritrack_model():
    base  = os.path.dirname(__file__)
    model = load_model(
        os.path.join(base, '..', 'model', 'nutritrack_best_model.keras'),
        custom_objects={
            'AttentionLayer': AttentionLayer,
            'FocalLoss'     : FocalLoss
        }
    )
    return model

@st.cache_data
def load_data():
    base   = os.path.dirname(__file__)
    os.path.join(base, '..', 'data', 'processed', 'dataset_cleaned.csv')
    os.path.join(base, '..', 'data', 'raw', 'ObesityDataSet_raw_and_data_sinthetic__1_.csv')
    os.path.join(base, '..', 'data', 'processed', 'X_test.npy')
    os.path.join(base, '..', 'data', 'processed', 'y_test.npy')
    return df, df_raw, X_test, y_test

# ============================================================
# HEADER
# ============================================================
st.title("🥗 NutriTrack AI")
st.markdown("#### Sistem Prediksi & Rekomendasi Level Obesitas berbasis Deep Learning")
st.markdown("---")

# Load semua resource
try:
    model                        = load_nutritrack_model()
    df, df_raw, X_test, y_test   = load_data()
    df_raw_clean                 = df_raw.drop_duplicates().copy()
    df_raw_clean['BMI']          = df_raw_clean['Weight'] / (df_raw_clean['Height'] ** 2)
    model_loaded                 = True
except Exception as e:
    st.error(f"⚠️ Gagal load model/data: {e}")
    st.info("Pastikan file berikut ada di folder yang sama:\n- nutritrack_best_model.keras\n- dataset_cleaned.csv\n- ObesityDataSet_raw_and_data_sinthetic__1_.csv\n- X_test.npy, y_test.npy")
    model_loaded = False

# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard EDA",
    "🔍 Prediksi Obesitas",
    "📈 Performa Model"
])

# ============================================================
# TAB 1 — DASHBOARD EDA
# ============================================================
with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Insight dari dataset obesitas yang digunakan untuk melatih model NutriTrack.")

    if model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data",     f"{len(df_raw):,} baris")
        col2.metric("Jumlah Fitur",   "16 fitur")
        col3.metric("Jumlah Kelas",   "7 kelas")
        col4.metric("Missing Values", "0 ✅")

        st.markdown("---")

        order_bmi = [
            'Insufficient_Weight', 'Normal_Weight',
            'Overweight_Level_I',  'Overweight_Level_II',
            'Obesity_Type_I',      'Obesity_Type_II', 'Obesity_Type_III'
        ]

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribusi Kelas Obesitas")
            fig, ax = plt.subplots(figsize=(6, 4))
            order  = df['obesity_level'].value_counts().index
            colors = sns.color_palette("Set2", len(order))
            df['obesity_level'].value_counts()[order].plot(
                kind='bar', ax=ax, color=colors, edgecolor='white'
            )
            ax.set_xlabel('Kelas Obesitas')
            ax.set_ylabel('Jumlah')
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', fontsize=9)
            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("💡 Distribusi kelas cukup seimbang (272–351 per kelas)")

        with col_b:
            st.subheader("Distribusi BMI per Kelas")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_raw_clean, x='NObeyesdad', y='BMI',
                        order=order_bmi, palette='Set2', ax=ax)
            ax.set_xlabel('Kelas')
            ax.set_ylabel('BMI')
            plt.xticks(rotation=30, ha='right', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("💡 BMI meningkat konsisten seiring kelas obesitas")

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Aktivitas Fisik per Kelas")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_raw_clean, x='NObeyesdad', y='FAF',
                        order=order_bmi, palette='Set3', ax=ax)
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Frekuensi Aktivitas Fisik')
            plt.xticks(rotation=30, ha='right', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("💡 Penderita obesitas cenderung memiliki aktivitas fisik lebih rendah")

        with col_d:
            st.subheader("Riwayat Keluarga per Kelas")
            cross = pd.crosstab(
                df_raw_clean['NObeyesdad'],
                df_raw_clean['family_history_with_overweight'],
                normalize='index'
            ) * 100
            cross = cross.reindex(order_bmi)
            fig, ax = plt.subplots(figsize=(6, 4))
            cross.plot(kind='bar', ax=ax,
                       color=['#e07b7b', '#7bb8e0'], edgecolor='white')
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Persentase (%)')
            ax.legend(['Tidak ada riwayat', 'Ada riwayat'], fontsize=8)
            plt.xticks(rotation=30, ha='right', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("💡 Mayoritas Obesity Type III memiliki riwayat keluarga overweight")

# ============================================================
# TAB 2 — PREDIKSI
# ============================================================
with tab2:
    st.header("🔍 Prediksi Level Obesitas")
    st.markdown("Isi data di bawah ini untuk mendapatkan prediksi dan rekomendasi personal.")

    if model_loaded:
        with st.form("form_prediksi"):
            st.subheader("📝 Data Diri")
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
                age    = st.number_input("Usia (tahun)", 10, 80, 25)
                height = st.number_input("Tinggi Badan (cm)", 130, 220, 165)
            with col2:
                weight         = st.number_input("Berat Badan (kg)", 30, 200, 70)
                family_history = st.selectbox("Riwayat Keluarga Overweight", ["Tidak", "Ya"])
                high_cal_food  = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", ["Tidak", "Ya"])
            with col3:
                smoking        = st.selectbox("Merokok?", ["Tidak", "Ya"])
                cal_monitoring = st.selectbox("Pantau Kalori Harian?", ["Tidak", "Ya"])
                alcohol        = st.selectbox("Konsumsi Alkohol", ["Tidak", "Kadang", "Sering", "Selalu"])

            st.subheader("🍽️ Kebiasaan Makan & Aktivitas")
            col4, col5, col6 = st.columns(3)
            with col4:
                veg_freq  = st.slider("Frekuensi Makan Sayur (1=Jarang, 5=Selalu)", 1, 5, 2)
                meals_day = st.slider("Jumlah Makan Utama per Hari", 1, 6, 3)
            with col5:
                snack = st.selectbox("Ngemil di Luar Makan Utama", ["Tidak", "Kadang", "Sering", "Selalu"])
                water = st.slider("Konsumsi Air per Hari (liter)", 1, 5, 2)
            with col6:
                faf       = st.slider("Frekuensi Olahraga per Minggu (hari)", 0, 7, 1)
                tue       = st.slider("Waktu Pakai Teknologi per Hari (jam)", 0, 12, 4)
                transport = st.selectbox("Transportasi Utama",
                                         ["Mobil", "Sepeda", "Motor",
                                          "Transportasi Umum", "Jalan Kaki"])

            submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

        if submitted:
            gender_enc    = 1 if gender == "Male" else 0
            fam_hist_enc  = 1 if family_history == "Ya" else 0
            high_cal_enc  = 1 if high_cal_food == "Ya" else 0
            smoking_enc   = 1 if smoking == "Ya" else 0
            cal_mon_enc   = 1 if cal_monitoring == "Ya" else 0
            alcohol_enc   = {"Tidak": 0, "Kadang": 1, "Sering": 2, "Selalu": 3}[alcohol]
            snack_enc     = {"Tidak": 0, "Kadang": 1, "Sering": 2, "Selalu": 3}[snack]
            transport_enc = {"Mobil": 0, "Sepeda": 1, "Motor": 2,
                             "Transportasi Umum": 3, "Jalan Kaki": 4}[transport]

            def scale(val, min_val, max_val):
                return (val - min_val) / (max_val - min_val)

            height_m   = height / 100
            input_data = [
                gender_enc,
                scale(age,      14, 61),
                scale(height_m, 1.45, 1.98),
                scale(weight,   39, 173),
                fam_hist_enc,
                high_cal_enc,
                scale(veg_freq,  1, 5),
                scale(meals_day, 1, 6),
                snack_enc,
                smoking_enc,
                scale(water, 1, 5),
                cal_mon_enc,
                scale(faf, 0, 7),
                scale(tue, 0, 12),
                alcohol_enc,
                transport_enc,
            ]

            input_array = np.array(input_data).reshape(1, -1)
            proba       = model.predict(input_array, verbose=0)[0]
            pred_idx    = np.argmax(proba)
            pred_name   = TARGET_NAMES[pred_idx]
            confidence  = proba[pred_idx]
            rekomen     = REKOMENDASI[pred_name]

            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown(f"""
                <div style='background-color:{rekomen["warna"]}22;
                            border-left: 5px solid {rekomen["warna"]};
                            padding: 20px; border-radius: 10px;'>
                    <h3 style='color:{rekomen["warna"]}'>{rekomen["status"]}</h3>
                    <h4>{pred_name.replace("_", " ")}</h4>
                    <p>Confidence: <b>{confidence*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                st.markdown(f"**📝 {rekomen['deskripsi']}**")
                st.markdown("**Probabilitas semua kelas:**")
                prob_df = pd.DataFrame({
                    'Kelas'       : [n.replace('_', ' ') for n in TARGET_NAMES],
                    'Probabilitas': proba
                }).sort_values('Probabilitas', ascending=True)
                fig, ax    = plt.subplots(figsize=(6, 3))
                colors_bar = ['#DC3545' if n.replace(' ', '_') == pred_name
                              else '#6c757d' for n in prob_df['Kelas']]
                ax.barh(prob_df['Kelas'], prob_df['Probabilitas'],
                        color=colors_bar, edgecolor='white')
                ax.set_xlabel('Probabilitas')
                ax.set_xlim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("💡 Rekomendasi Personal")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.markdown("#### 🥗 Diet")
                for r in rekomen['diet']:
                    st.markdown(f"• {r}")
            with col_r2:
                st.markdown("#### 🏃 Olahraga")
                for r in rekomen['olahraga']:
                    st.markdown(f"• {r}")
            with col_r3:
                st.markdown("#### 🏥 Medis")
                for r in rekomen['medis']:
                    st.markdown(f"• {r}")

            st.info("⚠️ Rekomendasi ini bersifat informatif. Selalu konsultasikan kondisi kesehatanmu dengan dokter atau ahli gizi profesional.")

# ============================================================
# TAB 3 — PERFORMA MODEL
# ============================================================
with tab3:
    st.header("📈 Performa Model NutriTrack")

    if model_loaded:
        y_pred_proba        = model.predict(X_test, verbose=0)
        y_pred              = np.argmax(y_pred_proba, axis=1)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy",    f"{test_acc*100:.2f}%",
                    delta="✅ ≥ 85%" if test_acc >= 0.85 else "⚠️ < 85%")
        col2.metric("Test Loss",        f"{test_loss:.4f}")
        col3.metric("Jumlah Test Data", f"{len(y_test)} sampel")

        st.markdown("---")

        st.subheader("📋 Classification Report")
        report    = classification_report(
            y_test, y_pred,
            target_names=[n.replace('_', ' ') for n in TARGET_NAMES],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🔲 Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[n.replace('_', ' ') for n in TARGET_NAMES],
            yticklabels=[n.replace('_', ' ') for n in TARGET_NAMES],
            ax=ax, linewidths=0.5
        )
        ax.set_title('Confusion Matrix — NutriTrack Model', fontsize=13, fontweight='bold')
        ax.set_xlabel('Prediksi', fontsize=11)
        ax.set_ylabel('Aktual', fontsize=11)
        plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        try:
            base   = os.path.dirname(__file__)
            log_df = pd.read_csv(os.path.join(base, '..', 'logs', 'training_log.csv'))
            st.subheader("📉 Kurva Training")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(log_df['epoch'], log_df['accuracy'],
                         label='Train', color='steelblue', linewidth=2)
            axes[0].plot(log_df['epoch'], log_df['val_accuracy'],
                         label='Validation', color='coral', linewidth=2)
            axes[0].set_title('Accuracy per Epoch')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[1].plot(log_df['epoch'], log_df['loss'],
                         label='Train', color='steelblue', linewidth=2)
            axes[1].plot(log_df['epoch'], log_df['val_loss'],
                         label='Validation', color='coral', linewidth=2)
            axes[1].set_title('Loss per Epoch')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)
        except:
            st.info("File training_log.csv tidak ditemukan.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "🥗 NutriTrack AI — Capstone Project Data Science | "
    "Model: TensorFlow Functional API + Custom Components"
    "</div>",
    unsafe_allow_html=True
)
