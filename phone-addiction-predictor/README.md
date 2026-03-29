# 📱 Phone Addiction Level Predictor

Aplikasi web berbasis **Streamlit** untuk memprediksi tingkat kecanduan smartphone
menggunakan model **CatBoost Regressor** yang dilatih pada dataset `Phone_Addiction.csv`.

| Metrik | Nilai |
|--------|-------|
| RMSE (test) | 0.3667 |
| R² (test) | 0.9453 |

---

## 📁 Struktur Proyek

```
phone-addiction-predictor/
├── app.py                  # Aplikasi Streamlit utama
├── train_and_save.py       # Script training & simpan artifact
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Pipeline preprocessing
│   └── model.py            # Load artifact & inferensi
├── models/
│   ├── catboost_model.cbm  # Model CatBoost (native format)
│   ├── scaler.pkl          # StandardScaler (joblib)
│   └── encoders.pkl        # OHE + medians + modes + feature_order
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Instalasi

### 1. Clone / download proyek

```bash
git clone <repo-url>
cd phone-addiction-predictor
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Model

Jalankan script berikut **sekali** untuk melatih model dan menyimpan semua artifact ke folder `models/`:

```bash
python train_and_save.py
```

Output yang diharapkan:

```
Loading dataset...
Cleaning data...
...
=== Test Results ===
RMSE : 0.3667
R²   : 0.9453

Saving artifacts...
  Saved: models/catboost_model.cbm
  Saved: models/scaler.pkl
  Saved: models/encoders.pkl

Done! You can now run: streamlit run app.py
```

> **Catatan:** Pastikan file `Phone_Addiction.csv` ada di folder **parent** dari `phone-addiction-predictor/`
> (yaitu satu level di atas, sejajar dengan folder proyek ini).

---

## 🚀 Menjalankan Aplikasi

```bash
# Dari root workspace (satu level di atas phone-addiction-predictor/)
python -m streamlit run phone-addiction-predictor/app.py

# Atau dari dalam folder phone-addiction-predictor/
python -m streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 🎯 Cara Penggunaan

1. Isi semua field input yang tersedia (19 fitur)
2. Klik tombol **🔍 Prediksi**
3. Lihat hasil prediksi beserta interpretasinya

### Seksi Input

| Seksi | Fitur |
|-------|-------|
| 👤 Informasi Dasar | Age, Gender, Daily_Usage_Hours, Sleep_Hours, Weekend_Usage_Hours |
| 📲 Aktivitas Smartphone | Phone_Checks_Per_Day, Apps_Used_Daily, Screen_Time_Before_Bed, Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose |
| 🧠 Kesehatan & Sosial | Anxiety_Level, Depression_Level, Self_Esteem, Interllectual_Performance, Social_Interactions, Exercise_Hours, Family_Communication |

### Interpretasi Hasil

| Nilai | Kategori | Keterangan |
|-------|----------|------------|
| 1.0 – 3.9 | 🟢 Rendah | Penggunaan smartphone tergolong sehat |
| 4.0 – 6.9 | 🟡 Sedang | Perlu perhatian pada pola penggunaan |
| 7.0 – 10.0 | 🔴 Tinggi | Disarankan mengurangi penggunaan smartphone |

---

## 🔧 Pipeline Preprocessing

Pipeline di aplikasi **identik** dengan notebook pelatihan:

1. **Clean** — strip kutip dari `Sleep_Hours`, normalisasi `Gender`
2. **Impute** — median untuk numerik, modus untuk kategorikal (dari training set)
3. **OHE** — `Gender` & `Phone_Usage_Purpose` dengan `drop=["Other","Other"]`
4. **Feature Engineering** — 10 fitur turunan (ratio, interaksi, flag)
5. **Log Transform** — `np.log1p` pada 7 kolom skewed
6. **Scale** — `StandardScaler` yang di-fit pada training set

---

## 📦 Dependencies

```
streamlit>=1.28.0
catboost>=1.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```
