"""
prediction.py
==============
Logika prediksi inti untuk FINTel Customer Churn Intelligence Dashboard.

Mereplikasi alur dari notebook:
  - Segmentasi Cold Start (tenure <= 5 bulan) vs Non-Cold Start (tenure > 5 bulan)
  - Pemilihan model + threshold sesuai segmen
  - Piecewise probability -> Churn Score (0-100) scaler
  - Klasifikasi Risk Zone (AMAN / AMBIGU / TINGGI)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# Konstanta (identik dengan notebook, sel "Piecewise Churn Score Scaler"
# dan "Churn Risk Dashboard")
# ──────────────────────────────────────────────────────────────────────────
CHURNSCORE_CRITICAL = 54   # titik kritis EDA = batas bawah Gray Area
CHURNSCORE_MAX = 100
CHURNSCORE_MIN = 0

GRAY_ZONE_LOWER = 54
GRAY_ZONE_UPPER = 80

COLD_START_TENURE_CUTOFF = 5

ADDON_COLUMNS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# Kolom yang dipakai sebagai fitur model (urutan tidak penting karena
# ColumnTransformer mengenali lewat nama kolom, tapi kolom ini WAJIB
# tersedia persis seperti saat training / split_data()).
#
# PENTING: split_data() pada notebook TIDAK men-drop kolom 'CLTV', sehingga
# CLTV ikut masuk sebagai fitur model lewat ColumnTransformer(remainder=
# 'passthrough'). Kolom ini harus disertakan di sini agar input prediksi
# konsisten dengan data training.
MODEL_FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'CLTV',
]


@dataclass
class CustomerPrediction:
    customer_id: str
    segment: str                  # 'Cold Start' | 'Non-Cold Start'
    is_cold_start: bool
    churn_probability: float      # 0-1, output model
    churn_score: float            # 0-100, hasil piecewise scaler
    threshold: float
    is_churn_predicted: bool
    risk_zone: str                # 'AMAN' | 'AMBIGU' | 'TINGGI'
    status_label: str             # 'BERISIKO CHURN' | 'AMAN'
    top_shap_features: list = field(default_factory=list)   # list[(feature, value)]
    risk_profile: str = ""
    risk_reason: str = ""
    recommendations: list = field(default_factory=list)
    customer_raw: dict = field(default_factory=dict)
    explainer_type: str = ""


def prob_to_churnscore(prob, threshold: float) -> float:
    """
    Petakan probability model (0-1) ke skala Churn Score (0-100) secara
    piecewise linear, identik dengan fungsi `prob_to_churnscore` di notebook.

    3 titik anchor:
      prob = 0          -> score = 0
      prob = threshold   -> score = 54   (titik keputusan model)
      prob = 1          -> score = 100
    """
    prob = np.asarray(prob, dtype=float)
    thr = min(max(threshold, 1e-6), 1 - 1e-6)

    below = prob <= thr
    score = np.empty_like(prob, dtype=float)
    score[below] = CHURNSCORE_MIN + (prob[below] - 0.0) / (thr - 0.0) * (CHURNSCORE_CRITICAL - CHURNSCORE_MIN)
    score[~below] = CHURNSCORE_CRITICAL + (prob[~below] - thr) / (1.0 - thr) * (CHURNSCORE_MAX - CHURNSCORE_CRITICAL)
    score = np.clip(score, CHURNSCORE_MIN, CHURNSCORE_MAX)

    return float(score) if score.shape == () else score


def get_risk_zone(churn_score: float) -> str:
    """
    - 'AMAN'    : skor < 54
    - 'AMBIGU'  : 54 <= skor <= 80
    - 'TINGGI'  : skor > 80
    """
    if churn_score < GRAY_ZONE_LOWER:
        return 'AMAN'
    elif churn_score <= GRAY_ZONE_UPPER:
        return 'AMBIGU'
    else:
        return 'TINGGI'


def determine_segment(tenure: float) -> str:
    """Tentukan segmen Cold Start / Non-Cold Start berdasarkan tenure (bulan)."""
    return 'Cold Start' if tenure <= COLD_START_TENURE_CUTOFF else 'Non-Cold Start'


def compute_add_on_count(customer_row: pd.Series) -> int:
    """Hitung jumlah layanan tambahan (add-on) yang dimiliki pelanggan."""
    return int(sum(1 for col in ADDON_COLUMNS if customer_row.get(col) == 'Yes'))


def find_customer(customer_id: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Cari satu baris customer berdasarkan customerID (case-insensitive, trim spasi)."""
    if df is None or df.empty:
        return None
    customer_id = str(customer_id).strip()
    matches = df[df['customerID'].astype(str).str.strip().str.upper() == customer_id.upper()]
    if matches.empty:
        return None
    return matches.iloc[0]


def prepare_model_input(customer_row: pd.Series) -> pd.DataFrame:
    """
    Susun satu baris DataFrame berisi fitur mentah yang sesuai urutan kolom
    yang dipakai saat training (sebelum masuk ColumnTransformer pipeline).
    Pipeline (preprocessor) yang akan melakukan encoding + scaling sendiri.
    """
    row = {col: customer_row.get(col) for col in MODEL_FEATURE_COLUMNS}
    return pd.DataFrame([row])


def predict_customer(customer_row: pd.Series, cold_bundle: dict, noncold_bundle: dict) -> dict:
    """
    Jalankan prediksi end-to-end untuk satu customer (tanpa SHAP/recommendation,
    itu ditangani modul terpisah). Return dict siap pakai oleh shap_utils &
    recommendation, serta oleh UI.
    """
    tenure = float(customer_row.get('tenure', 0) or 0)
    segment = determine_segment(tenure)
    is_cold_start = segment == 'Cold Start'

    bundle = cold_bundle if is_cold_start else noncold_bundle
    model = bundle['model']
    threshold = bundle['threshold']

    X_input = prepare_model_input(customer_row)
    churn_prob = float(model.predict_proba(X_input)[:, 1][0])
    churn_score = float(prob_to_churnscore(churn_prob, threshold))
    is_churn = churn_prob >= threshold
    risk_zone = get_risk_zone(churn_score)

    return {
        'segment': segment,
        'is_cold_start': is_cold_start,
        'churn_probability': churn_prob,
        'churn_score': churn_score,
        'threshold': threshold,
        'is_churn_predicted': is_churn,
        'risk_zone': risk_zone,
        'status_label': 'BERISIKO CHURN' if is_churn else 'AMAN',
        'X_input': X_input,
        'model_name': bundle.get('model_name', 'Unknown'),
    }
