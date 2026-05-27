# Project Flow and Specifications

This file aggregates the original specification documents from the `.kiro` folder, combining requirements, system design, and implementation tracking into a single unified reference.

---

## 1. Requirements Specification

### Introduction

Fitur ini mendeploy model Machine Learning CatBoost (RMSE test 0.362, R² test 0.947) yang telah dilatih pada dataset `Phone_Addiction.csv` ke dalam sebuah aplikasi web interaktif menggunakan Streamlit. Aplikasi menerima input perilaku penggunaan smartphone dari pengguna, menjalankan pipeline preprocessing yang identik dengan notebook pelatihan, lalu menampilkan prediksi tingkat kecanduan smartphone pada skala 1–10 beserta interpretasinya.

### Glossary

- **App**: Aplikasi Streamlit (`app.py`) yang menjadi antarmuka pengguna.
- **Preprocessor**: Modul `src/preprocessing.py` yang menjalankan seluruh langkah transformasi data.
- **Model**: Modul `src/model.py` yang memuat model CatBoost dan menjalankan inferensi.
- **Pipeline**: Urutan transformasi data yang harus identik antara training dan inferensi: cleaning → imputation → OHE → feature engineering → log transform → scaling.
- **Artifact**: File tersimpan hasil training: `models/catboost_model.cbm`, `models/scaler.pkl`, `models/encoders.pkl`.
- **Addiction_Level**: Target prediksi, nilai kontinu pada skala 1.0–10.0.
- **OHE**: OneHotEncoder untuk kolom `Gender` dan `Phone_Usage_Purpose`.
- **Scaler**: StandardScaler yang di-fit pada data training.
- **train_and_save.py**: Script Python untuk melatih ulang model dan menyimpan semua artifact.

---

### Requirements

#### Requirement 1: Setup Struktur Proyek

**User Story:** Sebagai developer, saya ingin struktur proyek yang terorganisir, sehingga kode mudah dipelihara dan di-deploy.

##### Acceptance Criteria

1. THE App SHALL memiliki struktur direktori: `phone-addiction-predictor/` dengan subfolder `src/` dan `models/`.
2. THE App SHALL menyertakan file `app.py`, `src/preprocessing.py`, `src/model.py`, `requirements.txt`, `README.md`, dan `.gitignore` pada direktori root proyek.
3. THE App SHALL menyertakan file `train_and_save.py` pada direktori root untuk keperluan training ulang dan penyimpanan artifact.
4. WHEN folder `models/` tidak berisi artifact, THEN THE App SHALL menampilkan pesan error yang informatif kepada pengguna.

---

#### Requirement 2: Preprocessing Pipeline yang Identik dengan Notebook

**User Story:** Sebagai data scientist, saya ingin pipeline preprocessing di aplikasi identik dengan notebook pelatihan, sehingga prediksi model valid dan konsisten.

##### Acceptance Criteria

1. THE Preprocessor SHALL membersihkan kolom `Sleep_Hours` dengan cara: konversi ke string, strip karakter kutip (`"`), lalu konversi ke float.
2. THE Preprocessor SHALL menormalisasi kolom `Gender` dengan cara: strip whitespace, lowercase, capitalize, dan mengganti nilai `"femle"` menjadi `"Female"`.
3. THE Preprocessor SHALL mengganti nilai `"Unknown"` pada kolom `Phone_Usage_Purpose` dengan `NaN` sebelum imputation.
4. THE Preprocessor SHALL melakukan imputation nilai numerik yang hilang menggunakan median dari data training.
5. THE Preprocessor SHALL melakukan imputation nilai kategorikal yang hilang menggunakan modus dari data training.
6. THE Preprocessor SHALL menerapkan OneHotEncoder dengan parameter `drop=["Other", "Other"]`, `sparse_output=False`, `handle_unknown="ignore"` pada kolom `["Gender", "Phone_Usage_Purpose"]`.
7. THE Preprocessor SHALL membuat 11 fitur turunan (engineered features) sesuai formula notebook:
   - `usage_zero_flag`, `checks_per_hour`, `apps_per_hour`, `screen_before_bed_ratio`
   - `usage_to_sleep_ratio`, `late_screen_ratio`, `social_to_solo_ratio`
   - `resilience_gap`, `high_gaming_x_sleep`, `social_media_x_anxiety`
8. THE Preprocessor SHALL menerapkan transformasi `np.log1p` pada kolom skewed: `["Age", "checks_per_hour", "apps_per_hour", "screen_before_bed_ratio", "usage_to_sleep_ratio", "social_to_solo_ratio", "social_media_x_anxiety"]`.
9. THE Preprocessor SHALL menerapkan StandardScaler yang telah di-fit pada data training untuk mentransformasi input inferensi.
10. WHEN input inferensi diterima, THE Preprocessor SHALL menjalankan seluruh langkah 1–9 secara berurutan dalam satu fungsi `preprocess_pipeline()`.
11. FOR ALL input valid yang diproses oleh Preprocessor, urutan kolom output SHALL identik dengan urutan kolom saat training.

---

#### Requirement 3: Penyimpanan dan Pemuatan Artifact Model

**User Story:** Sebagai developer, saya ingin artifact model disimpan dan dimuat dengan benar, sehingga aplikasi dapat berjalan tanpa perlu melatih ulang setiap kali dijalankan.

##### Acceptance Criteria

1. THE train_and_save.py SHALL melatih model CatBoost menggunakan dataset `Phone_Addiction.csv` dengan parameter terbaik dari notebook.
2. THE train_and_save.py SHALL menyimpan model CatBoost ke `models/catboost_model.cbm` menggunakan metode native CatBoost.
3. THE train_and_save.py SHALL menyimpan objek `StandardScaler` yang telah di-fit ke `models/scaler.pkl` menggunakan `joblib`.
4. THE train_and_save.py SHALL menyimpan objek `OneHotEncoder` yang telah di-fit ke `models/encoders.pkl` menggunakan `joblib`.
5. THE Model SHALL memuat `catboost_model.cbm`, `scaler.pkl`, dan `encoders.pkl` dari folder `models/` saat aplikasi diinisialisasi.
6. WHEN file artifact tidak ditemukan, THEN THE Model SHALL melempar `FileNotFoundError` dengan pesan yang menyebutkan nama file yang hilang.
7. THE Model SHALL menggunakan `@st.cache_resource` untuk meng-cache pemuatan artifact agar tidak dimuat ulang setiap prediksi.

---

#### Requirement 4: Antarmuka Input Pengguna

**User Story:** Sebagai pengguna, saya ingin mengisi form input yang jelas dan terstruktur, sehingga saya dapat memasukkan data perilaku smartphone saya dengan mudah.

##### Acceptance Criteria

1. THE App SHALL menampilkan judul aplikasi dan deskripsi singkat pada halaman utama.
2. THE App SHALL menyediakan input untuk 19 fitur berikut:
   - `Age` (number input, min 1, max 100)
   - `Gender` (selectbox: Male, Female, Other)
   - `Daily_Usage_Hours` (number input, min 0.0, max 24.0)
   - `Sleep_Hours` (number input, min 0.0, max 24.0)
   - `Interllectual_Performance` (number input, min 0, max 100)
   - `Social_Interactions` (number input, min 0, max 20)
   - `Exercise_Hours` (number input, min 0.0, max 24.0)
   - `Screen_Time_Before_Bed` (number input, min 0.0, max 24.0)
   - `Phone_Checks_Per_Day` (number input, min 0, max 500)
   - `Anxiety_Level` (number input, min 0, max 10)
   - `Depression_Level` (number input, min 0, max 10)
   - `Self_Esteem` (number input, min 0, max 10)
   - `Apps_Used_Daily` (number input, min 0, max 100)
   - `Time_on_Social_Media` (number input, min 0.0, max 24.0)
   - `Time_on_Gaming` (number input, min 0.0, max 24.0)
   - `Time_on_Education` (number input, min 0.0, max 24.0)
   - `Phone_Usage_Purpose` (selectbox: Browsing, Education, Gaming, Social Media, Other)
   - `Family_Communication` (number input, min 0, max 20)
   - `Weekend_Usage_Hours` (number input, min 0.0, max 24.0)
3. THE App SHALL mengelompokkan input ke dalam beberapa seksi yang logis (misalnya: Informasi Dasar, Penggunaan Smartphone, Kesehatan Mental).
4. THE App SHALL menyediakan tombol "Prediksi" untuk memicu proses prediksi.
5. WHEN pengguna mengklik tombol "Prediksi", THE App SHALL menjalankan preprocessing dan inferensi model.

---

#### Requirement 5: Prediksi dan Tampilan Hasil

**User Story:** Sebagai pengguna, saya ingin melihat hasil prediksi tingkat kecanduan beserta interpretasinya, sehingga saya dapat memahami kondisi saya.

##### Acceptance Criteria

1. THE Model SHALL menerima DataFrame satu baris hasil preprocessing dan mengembalikan nilai prediksi `Addiction_Level` bertipe float.
2. THE App SHALL menampilkan nilai prediksi dibulatkan ke dua desimal pada skala 1.0–10.0.
3. THE App SHALL menampilkan interpretasi kategorikal berdasarkan nilai prediksi:
   - Nilai < 4.0: "Rendah – Penggunaan smartphone Anda tergolong sehat."
   - Nilai 4.0–6.9: "Sedang – Perhatikan pola penggunaan smartphone Anda."
   - Nilai ≥ 7.0: "Tinggi – Disarankan untuk mengurangi penggunaan smartphone."
4. THE App SHALL menampilkan indikator visual (misalnya progress bar atau warna) yang mencerminkan tingkat kecanduan.
5. IF terjadi error saat preprocessing atau inferensi, THEN THE App SHALL menampilkan pesan error yang informatif tanpa crash.

---

#### Requirement 6: Dependencies dan Konfigurasi Deployment

**User Story:** Sebagai developer, saya ingin semua dependency terdokumentasi dengan versi yang tepat, sehingga aplikasi dapat direproduksi di lingkungan lain.

##### Acceptance Criteria

1. THE App SHALL menyertakan `requirements.txt` yang mencantumkan semua dependency dengan versi yang kompatibel: `streamlit`, `catboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`.
2. THE App SHALL menyertakan `.gitignore` yang mengecualikan: folder `__pycache__/`, file `*.pyc`, folder `.venv/`, dan folder `models/*.cbm` (opsional, jika model tidak di-commit).
3. THE App SHALL menyertakan `README.md` dengan instruksi: instalasi dependency, cara menjalankan `train_and_save.py`, dan cara menjalankan aplikasi Streamlit.
4. WHEN `requirements.txt` diinstal pada Python 3.9+, THE App SHALL dapat dijalankan tanpa error dependency.

---

## 2. System Design

### Overview

Aplikasi Streamlit untuk prediksi tingkat kecanduan smartphone menggunakan model CatBoost yang telah dilatih. Arsitektur terdiri dari tiga lapisan: antarmuka pengguna (`app.py`), preprocessing (`src/preprocessing.py`), dan inferensi model (`src/model.py`). Semua artifact ML (model, scaler, encoder) disimpan di folder `models/` dan dimuat sekali saat startup.

---

### Architecture

```
phone-addiction-predictor/
├── app.py                    # Streamlit UI + orchestration
├── train_and_save.py         # Script training ulang + simpan artifact
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Pipeline preprocessing (identik notebook)
│   └── model.py              # Load artifact + predict
├── models/
│   ├── catboost_model.cbm    # CatBoost native format
│   ├── scaler.pkl            # StandardScaler (joblib)
│   └── encoders.pkl          # OneHotEncoder (joblib)
├── requirements.txt
├── README.md
└── .gitignore
```

### Data Flow

```
User Input (19 raw features)
        ↓
  app.py: collect_input() → dict
        ↓
  preprocessing.preprocess_pipeline(input_dict, ohe, scaler)
    ├── clean_sleep_hours()
    ├── handle_missing_values()
    ├── encode_categorical(ohe)
    ├── engineer_features()
    ├── log_transform()
    └── scale_features(scaler)
        ↓
  model.predict(processed_df) → float
        ↓
  app.py: display_result(prediction)
```

---

### Component Design

#### `src/preprocessing.py`

Semua fungsi menerima dan mengembalikan `pd.DataFrame`. Fungsi `preprocess_pipeline` adalah entry point utama untuk inferensi.

```python
def clean_sleep_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Strip kutip dan konversi Sleep_Hours ke float."""
    df = df.copy()
    df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    return df

def handle_missing_values(df: pd.DataFrame, num_medians: dict, cat_modes: dict) -> pd.DataFrame:
    """Impute missing values menggunakan statistik dari training set."""
    df = df.copy()
    for col, val in num_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df

def encode_categorical(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Terapkan OHE pada Gender dan Phone_Usage_Purpose."""
    cat_cols = ["Gender", "Phone_Usage_Purpose"]
    encoded = ohe.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols)
    return pd.concat([df, encoded_df], axis=1)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buat 10 fitur turunan sesuai notebook."""
    x = df.copy()
    eps = 1e-3
    x["usage_zero_flag"] = (x["Daily_Usage_Hours"] <= 0).astype(int)
    denom_usage = x["Daily_Usage_Hours"].clip(lower=1)
    x["checks_per_hour"] = x["Phone_Checks_Per_Day"] / denom_usage
    x["apps_per_hour"] = x["Apps_Used_Daily"] / denom_usage
    x["screen_before_bed_ratio"] = x["Screen_Time_Before_Bed"] / denom_usage
    x["usage_to_sleep_ratio"] = x["Daily_Usage_Hours"] / (x["Sleep_Hours"] + eps)
    x["late_screen_ratio"] = x["Screen_Time_Before_Bed"] / (x["Sleep_Hours"] + eps)
    solo_usage = x["Time_on_Gaming"] + x["Time_on_Social_Media"]
    social_use = x["Family_Communication"] + x["Social_Interactions"]
    x["social_to_solo_ratio"] = social_use / (solo_usage + eps)
    mental_strain = x["Anxiety_Level"] + x["Depression_Level"]
    x["resilience_gap"] = x["Self_Esteem"] - mental_strain / 2.0
    x["high_gaming_x_sleep"] = x["Time_on_Gaming"] * x["Sleep_Hours"]
    x["social_media_x_anxiety"] = x["Time_on_Social_Media"] * x["Anxiety_Level"]
    return x

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan log1p pada kolom skewed."""
    skewed_cols = [
        "Age", "checks_per_hour", "apps_per_hour", "screen_before_bed_ratio",
        "usage_to_sleep_ratio", "social_to_solo_ratio", "social_media_x_anxiety"
    ]
    x = df.copy()
    for col in skewed_cols:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))
    return x

def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Terapkan StandardScaler yang sudah di-fit."""
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)

def preprocess_pipeline(
    input_dict: dict,
    ohe: OneHotEncoder,
    scaler: StandardScaler,
    num_medians: dict,
    cat_modes: dict,
    feature_order: list
) -> pd.DataFrame:
    """Entry point: raw input dict → scaled DataFrame siap inferensi."""
    df = pd.DataFrame([input_dict])
    df = clean_sleep_hours(df)
    df["Phone_Usage_Purpose"] = df["Phone_Usage_Purpose"].replace("Unknown", np.nan)
    df = handle_missing_values(df, num_medians, cat_modes)
    df = encode_categorical(df, ohe)
    df = engineer_features(df)
    df = log_transform(df)
    df = df[feature_order]  # pastikan urutan kolom identik
    df = scale_features(df, scaler)
    return df
```

#### `src/model.py`

```python
import joblib
from catboost import CatBoostRegressor
import streamlit as st

MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    """Muat semua artifact sekali saat startup."""
    model = CatBoostRegressor()
    model.load_model(f"{MODELS_DIR}/catboost_model.cbm")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    artifacts = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    # artifacts berisi: ohe, num_medians, cat_modes, feature_order
    return model, scaler, artifacts

def predict(model: CatBoostRegressor, processed_df) -> float:
    """Jalankan inferensi dan kembalikan nilai prediksi."""
    prediction = model.predict(processed_df)
    return float(prediction[0])
```

#### `train_and_save.py`

Script standalone yang mereproduksi pipeline training dari notebook dan menyimpan semua artifact.

```python
# Langkah-langkah:
# 1. Load Phone_Addiction.csv
# 2. Data cleaning (drop cols, fix Sleep_Hours, fix Gender, replace Unknown, cap Age, drop duplicates)
# 3. Split X/y dengan test_size=0.2, random_state=284091, stratify=y (binned)
# 4. Imputation (fit on X_train)
# 5. Drop duplicates X_train setelah imputation
# 6. OHE fit on X_train[cat_cols]
# 7. Feature engineering + log transform
# 8. Scaler fit on X_train
# 9. Train CatBoost dengan best params
# 10. Simpan: model.save_model("models/catboost_model.cbm")
#             joblib.dump(scaler, "models/scaler.pkl")
#             joblib.dump({ohe, num_medians, cat_modes, feature_order}, "models/encoders.pkl")
```

**CatBoost best params** (dari hasil Optuna di notebook):
```python
params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 1,
    "verbose": 0
}
```
> Catatan: Parameter eksak harus diambil dari output Optuna di notebook. Nilai di atas adalah placeholder yang perlu diverifikasi.

#### `app.py`

```python
# Layout:
# - st.title("Phone Addiction Level Predictor")
# - st.markdown(deskripsi singkat)
# - Form input dalam 3 kolom / expander:
#     Seksi 1: Informasi Dasar (Age, Gender, Daily_Usage_Hours, Sleep_Hours, Weekend_Usage_Hours)
#     Seksi 2: Aktivitas Smartphone (Phone_Checks_Per_Day, Apps_Used_Daily, Screen_Time_Before_Bed,
#               Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose)
#     Seksi 3: Kesehatan & Sosial (Anxiety_Level, Depression_Level, Self_Esteem,
#               Interllectual_Performance, Social_Interactions, Exercise_Hours, Family_Communication)
# - Tombol "Prediksi"
# - Hasil: st.metric, st.progress, st.info/warning/error berdasarkan kategori
```

---

### Data Models

#### Input Dict (raw, dari form Streamlit)

```python
{
    "Age": float,
    "Gender": str,                    # "Male" | "Female" | "Other"
    "Daily_Usage_Hours": float,
    "Sleep_Hours": float,             # akan di-clean oleh preprocessor
    "Interllectual_Performance": int,
    "Social_Interactions": int,
    "Exercise_Hours": float,
    "Screen_Time_Before_Bed": float,
    "Phone_Checks_Per_Day": int,
    "Anxiety_Level": int,
    "Depression_Level": int,
    "Self_Esteem": int,
    "Apps_Used_Daily": int,
    "Time_on_Social_Media": float,
    "Time_on_Gaming": float,
    "Time_on_Education": float,
    "Phone_Usage_Purpose": str,       # "Browsing" | "Education" | "Gaming" | "Social Media" | "Other"
    "Family_Communication": int,
    "Weekend_Usage_Hours": float
}
```

#### Artifact Bundle (`encoders.pkl`)

```python
{
    "ohe": OneHotEncoder,             # fitted OHE
    "num_medians": dict,              # {col: median_value}
    "cat_modes": dict,                # {col: mode_value}
    "feature_order": list[str]        # urutan kolom setelah semua transformasi
}
```

---

### Error Handling

| Kondisi | Penanganan |
|---|---|
| File artifact tidak ada | `FileNotFoundError` dengan nama file, ditangkap di `app.py` dengan `st.error()` |
| Input di luar range | Validasi di form Streamlit via `min_value`/`max_value` |
| Prediksi di luar [1, 10] | Clip hasil ke range [1.0, 10.0] sebelum ditampilkan |
| Error preprocessing | `try/except` di `app.py`, tampilkan `st.error()` |

---

### Correctness Properties

#### Property 1: Pipeline Idempotence
Menjalankan `preprocess_pipeline` dua kali pada input yang sama harus menghasilkan output yang identik (tidak ada side effect).

```python
result1 = preprocess_pipeline(input_dict, ohe, scaler, ...)
result2 = preprocess_pipeline(input_dict, ohe, scaler, ...)
assert result1.equals(result2)
```

#### Property 2: Output Shape Invariant
Output `preprocess_pipeline` harus selalu memiliki shape `(1, N)` di mana N adalah jumlah fitur yang diharapkan model.

```python
result = preprocess_pipeline(input_dict, ...)
assert result.shape[0] == 1
assert result.shape[1] == len(feature_order)
```

#### Property 3: Prediction Range
Nilai prediksi setelah clipping harus selalu berada dalam rentang [1.0, 10.0].

```python
pred = predict(model, processed_df)
clipped = max(1.0, min(10.0, pred))
assert 1.0 <= clipped <= 10.0
```

#### Property 4: Feature Order Consistency
Urutan kolom output preprocessor harus identik dengan `feature_order` yang disimpan saat training.

```python
result = preprocess_pipeline(input_dict, ...)
assert list(result.columns) == feature_order
```

#### Property 5: No NaN in Output
Output `preprocess_pipeline` tidak boleh mengandung nilai NaN (semua missing value sudah diimputasi).

```python
result = preprocess_pipeline(input_dict, ...)
assert not result.isnull().any().any()
```

---

## 3. Implementation Tasks and Progress

- [x] 1. Setup Struktur Proyek
  - [x] 1.1 Buat folder `phone-addiction-predictor/` dengan subfolder `src/` dan `models/`
  - [x] 1.2 Buat file kosong: `app.py`, `src/__init__.py`, `src/preprocessing.py`, `src/model.py`, `train_and_save.py`
  - [x] 1.3 Buat file placeholder: `requirements.txt`, `README.md`, `.gitignore`

- [x] 2. Ekstrak Preprocessing Code (`src/preprocessing.py`)
  - [x] 2.1 Implementasi `clean_sleep_hours(df)` — strip kutip, konversi ke float
  - [x] 2.2 Implementasi `handle_missing_values(df, num_medians, cat_modes)` — impute numerik dengan median, kategorikal dengan modus
  - [x] 2.3 Implementasi `encode_categorical(df, ohe)` — OHE untuk `Gender` dan `Phone_Usage_Purpose`
  - [x] 2.4 Implementasi `engineer_features(df)` — buat 10 fitur turunan sesuai notebook
  - [x] 2.5 Implementasi `log_transform(df)` — `np.log1p` pada 7 kolom skewed
  - [x] 2.6 Implementasi `scale_features(df, scaler)` — terapkan StandardScaler
  - [x] 2.7 Implementasi `preprocess_pipeline(input_dict, ohe, scaler, num_medians, cat_modes, feature_order)` — gabungkan semua langkah

- [x] 3. Ekstrak Training Code dan Simpan Artifact (`train_and_save.py` + `src/model.py`)
  - [x] 3.1 Implementasi `train_and_save.py`:
    - Load `Phone_Addiction.csv`
    - Data cleaning (drop cols, fix Sleep_Hours, fix Gender, replace Unknown→NaN, cap Age>150, drop duplicates)
    - Train/test split (`test_size=0.2, random_state=284091`)
    - Fit imputer (median numerik, modus kategorikal) pada X_train
    - Drop duplicates X_train setelah imputation
    - Fit OHE pada X_train[cat_cols] dengan `drop=["Other","Other"]`
    - Feature engineering + log transform pada X_train dan X_test
    - Fit StandardScaler pada X_train, transform keduanya
    - Train CatBoostRegressor dengan best params dari notebook
    - Simpan `models/catboost_model.cbm`, `models/scaler.pkl`, `models/encoders.pkl`
    - Print RMSE dan R² pada test set sebagai verifikasi
  - [x] 3.2 Implementasi `load_artifacts()` di `src/model.py` dengan `@st.cache_resource`
  - [x] 3.3 Implementasi `predict(model, processed_df)` di `src/model.py`

- [x] 4. Buat Aplikasi Streamlit (`app.py`)
  - [x] 4.1 Setup layout dasar: judul, deskripsi, import artifact via `load_artifacts()`
  - [x] 4.2 Implementasi form input 19 fitur dalam 3 seksi:
    - Seksi "Informasi Dasar": Age, Gender, Daily_Usage_Hours, Sleep_Hours, Weekend_Usage_Hours
    - Seksi "Aktivitas Smartphone": Phone_Checks_Per_Day, Apps_Used_Daily, Screen_Time_Before_Bed, Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose
    - Seksi "Kesehatan & Sosial": Anxiety_Level, Depression_Level, Self_Esteem, Interllectual_Performance, Social_Interactions, Exercise_Hours, Family_Communication
  - [x] 4.3 Hubungkan tombol "Prediksi" ke `preprocess_pipeline()` dan `predict()`
  - [x] 4.4 Tampilkan hasil prediksi: nilai numerik (2 desimal) + `st.progress` bar
  - [x] 4.5 Tampilkan interpretasi kategorikal:
    - < 4.0 → `st.success` "Rendah – Penggunaan smartphone Anda tergolong sehat."
    - 4.0–6.9 → `st.warning` "Sedang – Perhatikan pola penggunaan smartphone Anda."
    - ≥ 7.0 → `st.error` "Tinggi – Disarankan untuk mengurangi penggunaan smartphone."
  - [x] 4.6 Tambahkan error handling dengan `try/except` dan `st.error()` untuk kegagalan preprocessing/inferensi

- [x] 5. Setup Dependencies dan Dokumentasi
  - [x] 5.1 Tulis `requirements.txt` dengan versi yang kompatibel:
    ```
    streamlit>=1.28.0
    catboost>=1.2.0
    scikit-learn>=1.3.0
    pandas>=2.0.0
    numpy>=1.24.0
    joblib>=1.3.0
    ```
  - [x] 5.2 Tulis `.gitignore`:
    ```
    __pycache__/
    *.pyc
    .venv/
    *.egg-info/
    .DS_Store
    ```
  - [x] 5.3 Tulis `README.md` dengan:
    - Deskripsi proyek
    - Instruksi instalasi: `pip install -r requirements.txt`
    - Instruksi training: `python train_and_save.py`
    - Instruksi menjalankan app: `streamlit run app.py`
    - Penjelasan singkat fitur input dan output

- [x] 6. Testing dan Verifikasi
  - [x] 6.1 Jalankan `train_and_save.py` dan verifikasi RMSE ≈ 0.362, R² ≈ 0.947 pada test set
  - [x] 6.2 Jalankan `streamlit run app.py` and test dengan input default (nilai tengah dari range)
  - [x] 6.3 Test edge cases:
    - `Daily_Usage_Hours = 0` (usage_zero_flag = 1, denom_usage = 1)
    - `Sleep_Hours` sangat kecil (eps mencegah division by zero)
    - Input dengan nilai minimum dan maksimum semua fitur
    - [x] 6.4 Verifikasi urutan kolom output preprocessor identik dengan `feature_order` dari artifact
  - [x] 6.5 Perbaiki error yang ditemukan pada langkah 6.1–6.4
