"""
shap_utils.py
=============
Utilitas SHAP: menghitung SHAP value untuk satu customer dan memetakan nama
fitur hasil encoding (mis. "InternetService_0", "Contract_1") kembali ke
label yang mudah dibaca tim retensi.

Mereplikasi logika `build_shap_explainer`, `normalize_shap_2d`, dan bagian
"Top 3 faktor penentu prediksi" dari fungsi `predict_and_recommend()` di
notebook.
"""

import re
import numpy as np
import pandas as pd


def get_transformed_X(model_pipeline, X_input: pd.DataFrame) -> pd.DataFrame:
    """
    Jalankan seluruh step pipeline KECUALI 'classifier' dan 'resampler',
    identik dengan `get_transformed_X` di notebook. Hasilnya adalah fitur
    dalam representasi numerik (setelah encoding + scaling + feature
    selection) yang dipakai SHAP explainer.
    """
    X_transformed = X_input.copy()
    skip_steps = {'classifier', 'resampler'}
    for name, step in model_pipeline.steps:
        if name in skip_steps:
            continue
        X_transformed = step.transform(X_transformed)
    return X_transformed


def compute_shap_for_customer(model_pipeline, shap_bundle: dict, X_input: pd.DataFrame):
    """
    Hitung SHAP value untuk satu baris customer menggunakan explainer yang
    sudah dilatih (LinearExplainer / TreeExplainer / KernelExplainer, sesuai
    auto-dispatcher saat training).

    Returns
    -------
    shap_row : np.ndarray, shape (n_features,)
    feature_names : list[str]
    """
    explainer = shap_bundle['explainer']
    feature_names = shap_bundle['feature_names']

    X_trans = get_transformed_X(model_pipeline, X_input)
    if not isinstance(X_trans, pd.DataFrame):
        X_trans = pd.DataFrame(X_trans, columns=feature_names)

    shap_vals = explainer.shap_values(X_trans)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    sv = np.array(shap_vals)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    return sv[0], feature_names


def get_top_shap_features(shap_row: np.ndarray, feature_names: list, top_n: int = 3):
    """Ambil top-N fitur dengan |SHAP value| tertinggi, identik notebook."""
    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    return [(feature_names[i], float(shap_row[i])) for i in top_idx]


# ──────────────────────────────────────────────────────────────────────────
# Feature name mapping: encoded -> readable
# ──────────────────────────────────────────────────────────────────────────
# Hasil encoding pipeline notebook:
#  - OneHotEncoder(drop='first') pada: gender, SeniorCitizen, Partner,
#    Dependents, PhoneService, PaperlessBilling -> nama kolom seperti
#    "gender_Male", "SeniorCitizen_Yes", dst (readable secara native).
#  - BinaryEncoder pada: MultipleLines, InternetService, OnlineSecurity,
#    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
#    StreamingMovies, Contract, PaymentMethod -> nama kolom seperti
#    "InternetService_0", "Contract_1" (tidak readable secara native,
#    karena merupakan representasi biner dari urutan kategori).
#
# Mapping berikut memberi label deskriptif untuk kolom-kolom BinaryEncoder
# yang paling sering muncul sebagai top SHAP feature, supaya laporan ke tim
# retensi tetap mudah dibaca tanpa mengubah angka SHAP itu sendiri.
FEATURE_LABELS = {
    'gender_Male': 'Gender: Pria',
    'SeniorCitizen_Yes': 'Status Warga Senior',
    'Partner_Yes': 'Memiliki Pasangan',
    'Dependents_Yes': 'Memiliki Tanggungan',
    'PhoneService_Yes': 'Berlangganan Layanan Telepon',
    'PaperlessBilling_Yes': 'Paperless Billing (Tagihan Digital)',
    'tenure': 'Lama Berlangganan (Tenure)',
    'MonthlyCharges': 'Tagihan Bulanan',
    'TotalCharges': 'Total Tagihan',
}

# Pola umum untuk kolom hasil BinaryEncoder, mis. "InternetService_0"
_BINARY_ENCODED_BASE_LABELS = {
    'MultipleLines': 'Multiple Lines (Lini Telepon Ganda)',
    'InternetService': 'Jenis Layanan Internet',
    'OnlineSecurity': 'Online Security (Keamanan Online)',
    'OnlineBackup': 'Online Backup (Pencadangan Online)',
    'DeviceProtection': 'Device Protection (Perlindungan Perangkat)',
    'TechSupport': 'Tech Support (Dukungan Teknis)',
    'StreamingTV': 'Streaming TV',
    'StreamingMovies': 'Streaming Movies',
    'Contract': 'Jenis Kontrak',
    'PaymentMethod': 'Metode Pembayaran',
}

_binary_col_pattern = re.compile(r'^(' + '|'.join(_BINARY_ENCODED_BASE_LABELS.keys()) + r')_(\d+)$')


def humanize_feature_name(encoded_name: str) -> str:
    """
    Ubah nama fitur hasil encoding menjadi label yang mudah dibaca.
    Contoh: 'InternetService_0' -> 'Jenis Layanan Internet (komponen biner 0)'
            'tenure' -> 'Lama Berlangganan (Tenure)'
            'gender_Male' -> 'Gender: Pria'
    """
    if encoded_name in FEATURE_LABELS:
        return FEATURE_LABELS[encoded_name]

    match = _binary_col_pattern.match(encoded_name)
    if match:
        base, bit = match.group(1), match.group(2)
        base_label = _BINARY_ENCODED_BASE_LABELS[base]
        return f'{base_label}'

    # Fallback umum: ganti underscore jadi spasi & title-case
    return encoded_name.replace('_', ' ').strip()


def format_shap_features_for_display(top_features: list) -> list:
    """
    Format list (feature, shap_value) menjadi list dict siap pakai UI:
    [{'feature': ..., 'readable': ..., 'impact': ..., 'direction': ...}, ...]
    """
    formatted = []
    for feat, val in top_features:
        formatted.append({
            'feature': feat,
            'readable': humanize_feature_name(feat),
            'impact': val,
            'direction': 'mendorong churn' if val > 0 else 'menekan churn',
        })
    return formatted
