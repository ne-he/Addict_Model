# About the Project: Phone Addiction Level Predictor

Proyek ini adalah **Phone Addiction Level Predictor** (Prediktor Tingkat Kecanduan Smartphone). Proyek ini menggabungkan analisis data (Data Science), pemodelan pembelajaran mesin (Machine Learning), dan pengembangan aplikasi web interaktif (Web App) untuk memprediksi tingkat kecanduan seseorang terhadap ponselnya pada skala 1.0 hingga 10.0.

Berikut adalah penjelasan lengkap mengenai esensi, struktur, metrik, dan detail pemodelan proyek ini.

## 1. Esensi & Tujuan Proyek

Esensi dari proyek ini adalah memberikan penilaian objektif dan prediktif mengenai tingkat kecanduan smartphone seseorang berdasarkan kombinasi faktor:
- **Demografis**: Usia, jenis kelamin.
- **Pola Penggunaan Ponsel**: Durasi harian, durasi akhir pekan, jumlah cek ponsel, jumlah aplikasi yang digunakan.
- **Kesehatan Mental & Sosial**: Tingkat kecemasan, tingkat depresi, harga diri (self-esteem), interaksi sosial, jam olahraga, komunikasi dengan keluarga.

Dengan mengetahui estimasi skor kecanduan ini, pengguna dapat mengkategorikan kebiasaan mereka ke dalam tiga tingkat risiko:
- 🟢 **Rendah (1.0 – 3.9)**: Penggunaan sehat dan seimbang.
- 🟡 **Sedang (4.0 – 6.9)**: Mulai ada indikasi penggunaan berlebih, perlu pembatasan screen-time.
- 🔴 **Tinggi (7.0 – 10.0)**: Risiko kecanduan tinggi, sangat disarankan untuk mengurangi screen-time secara drastis atau berkonsultasi dengan profesional.

---

## 2. Struktur Proyek & File Penting

Di dalam workspace, proyek terbagi menjadi dua bagian utama:

### Analisis Eksploratif & Eksperimen Model
- [Phone_Addiction.csv](file:///c:/Users/wilhe/OneDrive/Documents/nemi/cv/prujek/End/Product/Addiction_predict/Addict_Model/Phone_Addiction.csv): Dataset utama berisi data profil pengguna dan label kecanduan (`Addiction_Level`).
- [AOL_Machine_Learning.ipynb](file:///c:/Users/wilhe/OneDrive/Documents/nemi/cv/prujek/End/Product/Addiction_predict/Addict_Model/AOL_Machine_Learning.ipynb): Jupyter Notebook tempat dilakukan analisis data eksploratif (EDA), pembersihan data, feature engineering, pencarian model terbaik, hyperparameter tuning, hingga evaluasi ensemble (Stacking).

### Aplikasi Web (Streamlit)
- [phone-addiction-predictor/app.py](file:///c:/Users/wilhe/OneDrive/Documents/nemi/cv/prujek/End/Product/Addiction_predict/Addict_Model/phone-addiction-predictor/app.py): Kode utama antarmuka pengguna (UI) berbasis Streamlit.
- [phone-addiction-predictor/train_and_save.py](file:///c:/Users/wilhe/OneDrive/Documents/nemi/cv/prujek/End/Product/Addiction_predict/Addict_Model/phone-addiction-predictor/train_and_save.py): Skrip otomatis untuk melatih model CatBoost terbaik dan menyimpan artefak model ke folder `models/`.
- [phone-addiction-predictor/src/preprocessing.py](file:///c:/Users/wilhe/OneDrive/Documents/nemi/cv/prujek/End/Product/Addiction_predict/Addict_Model/phone-addiction-predictor/src/preprocessing.py): Kode modular untuk memproses data mentah (baik saat latihan maupun saat input pengguna di web app) agar formatnya identik.

---

## 3. Variabel (Fitur) yang Digunakan

Model memprediksi kolom target `Addiction_Level` (skala 1–10) dengan menggunakan 19 fitur masukan berikut:

| Kategori | Nama Kolom / Fitur | Deskripsi |
| --- | --- | --- |
| **Profil & Waktu** | `Age`, `Gender`, `Sleep_Hours` | Usia, jenis kelamin, dan jam tidur harian |
| **Aktivitas Ponsel** | `Daily_Usage_Hours`, `Weekend_Usage_Hours`, `Phone_Checks_Per_Day`, `Apps_Used_Daily` | Durasi pakai harian/weekend, frekuensi mengecek HP, dan jumlah aplikasi aktif |
| **Tujuan & Fokus** | `Phone_Usage_Purpose`, `Time_on_Social_Media`, `Time_on_Gaming`, `Time_on_Education` | Kategori tujuan utama (Media Sosial, Game, dll) beserta durasinya masing-masing |
| **Mental & Fisik** | `Anxiety_Level`, `Depression_Level`, `Self_Esteem`, `Exercise_Hours` | Tingkat kecemasan & depresi (0-10), harga diri (0-10), dan durasi olahraga |
| **Sosial & Akademis** | `Social_Interactions`, `Family_Communication`, `Interllectual_Performance` | Jumlah interaksi sosial & komunikasi keluarga harian, serta performa akademis/intelektual |

---

## 4. Pipeline Preprocessing & Feature Engineering

Sebelum masuk ke model ML, data melalui serangkaian rekayasa fitur (feature engineering) untuk meningkatkan performa prediksi:
- **Pembersihan Data**: Menghapus data duplikat, memperbaiki format string pada kolom `Sleep_Hours`, dan menormalkan penulisan kategori `Gender`.
- **Imputasi Nilai Kosong**: Mengisi data numerik yang hilang menggunakan Median dan data kategorikal menggunakan Modus (berdasarkan data latih).
- **One-Hot Encoding**: Mengonversi kolom kategorikal `Gender` dan `Phone_Usage_Purpose` menjadi representasi biner.
- **Feature Engineering (10 Fitur Turunan)**:
  - `usage_zero_flag`: Menandai pengguna dengan screen time 0 jam.
  - `checks_per_hour`: Berapa kali mengecek HP tiap 1 jam pemakaian (`Phone_Checks_Per_Day` / `Daily_Usage_Hours`).
  - `apps_per_hour`: Jumlah aplikasi per jam pemakaian.
  - `screen_before_bed_ratio`: Rasio screen-time menjelang tidur dibanding total durasi harian.
  - `usage_to_sleep_ratio`: Rasio waktu pakai ponsel dibanding durasi tidur.
  - `late_screen_ratio`: Rasio waktu pakai ponsel sebelum tidur dibanding durasi tidur.
  - `social_to_solo_ratio`: Keseimbangan aktivitas sosial nyata dibanding aktivitas solo layar (Sosmed + Game).
  - `resilience_gap`: Skor kesehatan mental komposit (`Self_Esteem` dikurangi rata-rata kecemasan & depresi).
  - `high_gaming_x_sleep` & `social_media_x_anxiety`: Fitur interaksi perkalian antardua faktor penentu kecanduan.
- **Log Transformation (`np.log1p`)**: Mengurangi kemiringan (skewness) pada kolom berdistribusi tidak normal.
- **Standard Scaling**: Menyamakan skala semua fitur numerik agar performa algoritma berbasis gradien/jarak optimal.

---

## 5. Eksplorasi Model & Metrik Evaluasi

Pada notebook eksperimen, dilakukan perbandingan performa beberapa model Regresi. Metrik evaluasi yang digunakan adalah **RMSE** (Root Mean Squared Error) (semakin kecil mendekati 0 semakin baik) dan **R²** (R-squared / Koefisien Determinasi) (semakin mendekati 1.0 semakin baik).

Hasil evaluasi pada data uji (Test Set) menunjukkan performa sebagai berikut:
- **Decision Tree Regressor**: Mengalami overfitting yang cukup tinggi (RMSE uji lebih besar dibanding model boosting).
- **Random Forest Regressor**: Memberikan performa yang solid namun masih di bawah model boosting.
- **XGBoost & LightGBM Regressor**: Berkinerja sangat baik dengan RMSE uji di kisaran 0.36 – 0.39.
- **CatBoost Regressor (Model Terpilih)**:
  - Menghasilkan performa model tunggal terbaik.
  - **RMSE (Test)**: `0.3667` (artinya rata-rata simpangan prediksi model hanya sekitar ±0.37 dari skala kecanduan asli).
  - **R² Score (Test)**: `0.9453` (artinya 94.5% variasi tingkat kecanduan ponsel berhasil dijelaskan oleh fitur-fitur di dalam model).
- **Stacking Regressor (Ensemble)**:
  - Menggabungkan prediksi terbaik dari Decision Tree, Random Forest, XGBoost, LightGBM, dan CatBoost menggunakan LightGBM sebagai meta-learner.
  - Model Stacking memiliki RMSE dan R² yang setara dengan CatBoost (0.36 dan 0.95), tetapi memiliki kelebihan pada MAE (Mean Absolute Error) terkecil yaitu 0.13 (dibanding CatBoost sebesar 0.15). Artinya, jika Anda ingin meminimalkan deviasi kesalahan rata-rata absolut, model Stacking adalah opsi terbaik. Namun, demi kesederhanaan deployment and efisiensi waktu inferensi, aplikasi Streamlit Anda menggunakan model tunggal terbaik yaitu CatBoost Regressor.

---

## Repository GitHub

Proyek ini dapat diakses di repository GitHub berikut: [Addict_Model Repository](https://github.com/ne-he/Addict_Model.git)
