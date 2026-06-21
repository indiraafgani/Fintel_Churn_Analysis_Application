# FINTel Customer Churn Intelligence Dashboard

Aplikasi Streamlit untuk tim **Retention / Customer Success** FINTel. Masukkan
satu Customer ID, dapatkan laporan churn lengkap — skor risiko, profil
pelanggan, faktor pendorong churn (SHAP), dan rekomendasi aksi retensi —
setara dengan output fungsi `predict_and_recommend()` pada notebook riset
*Customer Churn Prediction: Solusi Data-Driven untuk Menekan Tingkat Churn di
FINTel*.

---

## 1. Struktur Proyek

```
project/
├── app.py                      # Entry point Streamlit (UI + orkestrasi)
├── train_model.py              # Script training — mereplikasi pipeline notebook
├── requirements.txt
├── README.md
├── df_full_with_id.csv         # Data training mentah (dipakai train_model.py)
├── models/
│   ├── cold_start_model.pkl    # Pipeline model + threshold, segmen Cold Start
│   ├── noncold_model.pkl       # Pipeline model + threshold, segmen Non-Cold Start
│   ├── cold_shap.pkl           # SHAP explainer, segmen Cold Start
│   ├── noncold_shap.pkl        # SHAP explainer, segmen Non-Cold Start
│   └── model_metadata.json     # Ringkasan metrik & konstanta (untuk sidebar app)
├── data/
│   └── customer_data.csv       # Basis data pelanggan yang dicari aplikasi
└── utils/
    ├── __init__.py
    ├── prediction.py           # Segmentasi, piecewise scaler, risk zone
    ├── recommendation.py       # Risk Profile Matcher + Recommendation Engine
    └── shap_utils.py           # Perhitungan SHAP + pemetaan nama fitur
```

---

## 2. Ringkasan Pendekatan Teknis

Aplikasi ini **bukan** re-implementasi yang disederhanakan — seluruh logika
inti notebook direplikasi persis:

| Komponen | Implementasi |
|---|---|
| Segmentasi | Cold Start (`tenure <= 5` bulan) vs Non-Cold Start (`tenure > 5` bulan), masing-masing dengan model & threshold terpisah |
| Preprocessing | `ColumnTransformer`: `OneHotEncoder(drop='first')` untuk 6 kolom kardinalitas rendah, `BinaryEncoder` (category_encoders) untuk 10 kolom kardinalitas sedang, `RobustScaler`, `SelectKBest` (ANOVA F-test) |
| Resampling | SMOTE / NeighbourhoodCleaningRule (dipilih otomatis saat benchmarking) |
| Model selection | Benchmark 11 algoritma (GridSearchCV, scoring FBeta β=2) → top-3 → `RandomizedSearchCV` tuning (100 iterasi/model) |
| Threshold | Dioptimasi per segmen via F2-score pada Precision-Recall curve (BUKAN default 0.5) |
| Churn Score (0–100) | Piecewise linear scaler: `prob=0→0`, `prob=threshold→54`, `prob=1→100` |
| Risk Zone | AMAN (`<54`) / AMBIGU (`54–80`) / TINGGI (`>80`) |
| Explainability | SHAP auto-dispatch: `LinearExplainer` (model linear) / `TreeExplainer` (tree-based) / `KernelExplainer` (fallback) |
| Risk Profile | Rule-based: Early Leaver, Underserved Customer, Price Sensitive, At-Risk Loyal |
| Recommendation | Playbook tetap per profil + rekomendasi prioritas tambahan untuk zona AMBIGU |

**Hasil training aktual** (lihat `models/model_metadata.json` untuk angka
presisi penuh):

| Segmen | Model | Threshold | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Cold Start | LogisticRegression | ≈0.2093 | 1.000 | 0.567 | 0.759 |
| Non-Cold Start | LogisticRegression | ≈0.5099 | 0.840 | 0.442 | 0.852 |

Angka ini identik dengan output notebook riset (dihasilkan dari data yang
sama: `df_full_with_id.csv`, derivasi dari `df_clean` notebook + `customerID`
hasil penggabungan ulang dengan dataset Kaggle Telco Customer Churn).

> **Catatan penting — kolom `CLTV`:** pada notebook, fungsi `split_data()`
> tidak mengecualikan kolom `CLTV` dari fitur model (hanya `ChurnScore`,
> `ChurnValue`, `ChurnReason`, dan kolom geografis yang dikecualikan). Akibatnya
> `CLTV` ikut masuk sebagai fitur lewat `ColumnTransformer(remainder=
> 'passthrough')`. Perilaku ini dipertahankan persis di `train_model.py` dan
> `utils/prediction.py` (`MODEL_FEATURE_COLUMNS`) agar hasil prediksi aplikasi
> konsisten dengan notebook.

---

## 3. Instalasi & Menjalankan Secara Lokal

```bash
# 1. Clone / masuk ke folder proyek
cd project

# 2. Buat virtual environment (sangat disarankan)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Opsional) Re-training model dari awal
#    Lewati langkah ini jika folder models/ & data/ sudah terisi.
python train_model.py

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

### Contoh Customer ID untuk uji coba
Setelah training, buka `data/customer_data.csv` dan ambil kolom `customerID`
mana saja — semua pelanggan dalam dataset bisa dicari. Aplikasi juga
menampilkan 10 contoh ID di layar awal sebelum pencarian dilakukan.

---

## 4. Re-training Model (`train_model.py`)

Jalankan `python train_model.py` jika ingin:
- Melatih ulang dengan data pelanggan terbaru (ganti isi `df_full_with_id.csv`
  dengan format kolom yang sama: seluruh kolom `df_clean` notebook + kolom
  `customerID`).
- Mengubah parameter (hyperparameter grid, threshold metric, dsb.) — edit
  langsung `train_model.py`, strukturnya mengikuti urutan sel notebook:
  `split_data` → `top_3model` (benchmarking) → `tuning` (RandomizedSearchCV)
  → `treshold` (F2 optimization) → SHAP explainer.

Script ini mencetak log per tahap (waktu fitting, skor benchmarking, threshold
optimal, classification report) dan pada akhirnya menyimpan ulang seluruh isi
folder `models/` dan `data/customer_data.csv`.

**Estimasi waktu training**: beberapa menit pada mesin multi-core
(`N_JOBS = 1` di script diset konservatif untuk lingkungan terbatas core;
naikkan ke `-1` pada mesin lokal/CI Anda untuk mempercepat).

---

## 5. Format Data Input (`df_full_with_id.csv`)

Kolom yang wajib ada (identik dengan `df_clean` + `customerID` pada notebook):

```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn, Country, State, City,
ZipCode, LatLong, Latitude, Longitude, ChurnValue, ChurnScore, CLTV, ChurnReason
```

---

## 6. Error Handling

| Kondisi | Perilaku Aplikasi |
|---|---|
| Customer ID tidak ditemukan | Tampilkan warning: *"Customer ID tidak ditemukan."* + daftar contoh ID |
| Model/data gagal dimuat | Tampilkan error eksplisit + detail file yang gagal, aplikasi berhenti dengan aman (`st.stop()`) |
| SHAP gagal dihitung untuk satu customer | Bagian "Top Churn Drivers" menampilkan info, bagian lain laporan tetap tampil |
| `customer_data.csv` kosong | Error eksplisit, aplikasi berhenti dengan aman |

---

## 7. Deployment ke GitHub

```bash
# 1. Inisialisasi repo (lewati jika sudah ada)
git init

# 2. Buat .gitignore agar folder venv/cache tidak ikut ter-commit
cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.DS_Store
EOF

# 3. Tambahkan seluruh file proyek
git add .
git commit -m "FINTel Customer Churn Intelligence Dashboard - initial release"

# 4. Hubungkan ke repo GitHub (buat repo kosong dulu di github.com)
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

> **Catatan ukuran file**: folder `models/` berisi file `.pkl` (~130–240 KB
> masing-masing) dan `data/customer_data.csv` (~2 MB) — semua jauh di bawah
> batas 100 MB GitHub, aman untuk di-push langsung tanpa Git LFS.

---

## 8. Deployment ke Streamlit Community Cloud

1. Pastikan repo GitHub di atas sudah berisi: `app.py`, `requirements.txt`,
   folder `models/`, `data/`, dan `utils/` (folder `models/` dan `data/`
   **wajib** ikut ter-push — aplikasi tidak melakukan training saat startup).
2. Buka **[share.streamlit.io](https://share.streamlit.io)** dan login dengan
   akun GitHub Anda.
3. Klik **"New app"**.
4. Pilih:
   - **Repository**: `<username>/<nama-repo>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. (Opsional) Klik **"Advanced settings"** jika perlu menetapkan Python
   version — gunakan Python 3.11 atau 3.12 agar kompatibel dengan versi
   library yang di-pin di `requirements.txt`.
6. Klik **"Deploy"**. Build pertama biasanya memakan waktu beberapa menit
   karena instalasi `scikit-learn`, `xgboost`, `lightgbm`, dan `shap`.
7. Setelah build selesai, aplikasi otomatis tersedia di URL
   `https://<nama-app>.streamlit.app` dan akan ter-redeploy otomatis setiap
   kali ada `git push` baru ke branch `main`.

### Troubleshooting Deployment
- **Build gagal karena versi library** — Streamlit Cloud menggunakan
  `requirements.txt` apa adanya; jika versi pinned tidak tersedia untuk
  platform Cloud, longgarkan pin tertentu (mis. ganti `==` jadi `>=`) lalu
  **wajib re-training ulang model** dengan versi library yang baru, karena
  pickle scikit-learn sensitif terhadap versi.
- **App crash saat load model** — cek log build/runtime di dashboard
  Streamlit Cloud; biasanya disebabkan oleh file `.pkl` yang corrupt saat
  push (pastikan tidak ter-track sebagai text oleh `.gitattributes`) atau
  versi `scikit-learn`/`imbalanced-learn` berbeda dari saat training.
- **Memory limit (Community Cloud free tier ≈1GB)** — model dalam aplikasi
  ini ringan (LogisticRegression, bukan ensemble besar), seharusnya jauh di
  bawah batas; jika tetap bermasalah, turunkan `customer_data.csv` ke subset
  data yang relevan saja.

---

## 9. Batasan & Pengembangan Lanjutan

- Risk Profile bersifat rule-based (bukan hasil clustering/model terpisah) —
  konsisten dengan desain notebook, namun bisa diperhalus dengan
  K-Means/DBSCAN di iterasi berikutnya.
- Model belum mempertimbangkan data temporal (survival analysis).
- Rekomendasi sebaiknya di-A/B test di lapangan untuk mengukur efektivitas
  aktual sebelum dijadikan playbook permanen.
- Untuk basis pelanggan yang berubah signifikan dari waktu ke waktu,
  jadwalkan re-training berkala (mis. tiap kuartal) via `train_model.py`.

---

## Referensi

Yulianti, Y., & Saifudin, A. (2020). *Sequential feature selection in customer
churn prediction based on Naive Bayes*. IOP Conference Series: Materials
Science and Engineering, 879, 012090.
https://doi.org/10.1088/1757-899X/879/1/012090

---

**Tim Penyusun (Notebook Riset Asli):** Akbar Kanugraha, Khaerun Nisa'Tri
Safaati, Indira Faisa Afgani — *Final Project, Data Science & Machine
Learning, Purwadhika Digital Technology School*.
