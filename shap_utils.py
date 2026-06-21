"""
report_builder.py
==================
Pipeline laporan churn terpadu untuk satu pelanggan — dipakai oleh ketiga
mode aplikasi (Existing Customer, New Customer manual input, Bulk Upload).

Memastikan setiap pelanggan, dari sumber input mana pun, melewati alur yang
PERSIS sama: segmentasi -> predict -> SHAP -> risk profile -> recommendation.
"""

from typing import Optional
import pandas as pd
import numpy as np

from utils.prediction import predict_customer, compute_add_on_count, MODEL_FEATURE_COLUMNS
from utils.shap_utils import (
    compute_shap_for_customer, get_top_shap_features, format_shap_features_for_display,
)
from utils.recommendation import assign_churn_profile, generate_recommendation, DEFAULT_MEDIAN_MONTHLY


def build_full_report(customer_row, cold_bundle: dict, noncold_bundle: dict,
                       cold_shap: dict, noncold_shap: dict,
                       median_monthly: float = DEFAULT_MEDIAN_MONTHLY) -> dict:
    """
    Jalankan pipeline prediksi end-to-end untuk satu customer dan return
    seluruh komponen laporan dalam satu dict siap pakai UI / export.

    Parameters
    ----------
    customer_row : pd.Series | dict
        Data mentah satu pelanggan. Wajib berisi kolom-kolom di
        MODEL_FEATURE_COLUMNS (lihat utils/prediction.py). Untuk pelanggan
        existing, ini biasanya satu baris dari customer_data.csv; untuk input
        manual / bulk upload, ini dibangun dari form input / file upload.
    cold_bundle, noncold_bundle : dict
        Hasil pickle.load() dari models/cold_start_model.pkl &
        models/noncold_model.pkl.
    cold_shap, noncold_shap : dict
        Hasil pickle.load() dari models/cold_shap.pkl & models/noncold_shap.pkl.
    median_monthly : float
        Ambang Price Sensitive (dari model_metadata.json).

    Returns
    -------
    dict berisi: segment, is_cold_start, churn_probability, churn_score,
        threshold, is_churn_predicted, risk_zone, status_label, model_name,
        shap_features (list of dict), risk_profile, risk_reason,
        recommendations (list), add_on_count, customer_raw (dict).
    """
    if isinstance(customer_row, dict):
        customer_row = pd.Series(customer_row)

    pred = predict_customer(customer_row, cold_bundle, noncold_bundle)

    model_pipeline = cold_bundle["model"] if pred["is_cold_start"] else noncold_bundle["model"]
    shap_bundle = cold_shap if pred["is_cold_start"] else noncold_shap

    try:
        shap_row, feature_names = compute_shap_for_customer(
            model_pipeline, shap_bundle, pred["X_input"]
        )
        top_shap = get_top_shap_features(shap_row, feature_names, top_n=3)
        shap_features = format_shap_features_for_display(top_shap)
    except Exception:
        shap_features = []

    addon_count = customer_row.get("add_on_count")
    if addon_count is None or (isinstance(addon_count, float) and np.isnan(addon_count)):
        addon_count = compute_add_on_count(customer_row)
    else:
        addon_count = int(addon_count)

    customer_raw = customer_row.to_dict() if hasattr(customer_row, "to_dict") else dict(customer_row)
    customer_raw["add_on_count"] = addon_count

    profile, reason = assign_churn_profile(
        customer_raw, is_cold_start=pred["is_cold_start"], median_monthly=median_monthly
    )
    recommendations = generate_recommendation(profile, risk_zone=pred["risk_zone"])

    return {
        "segment": pred["segment"],
        "is_cold_start": pred["is_cold_start"],
        "churn_probability": pred["churn_probability"],
        "churn_score": pred["churn_score"],
        "threshold": pred["threshold"],
        "is_churn_predicted": pred["is_churn_predicted"],
        "risk_zone": pred["risk_zone"],
        "status_label": pred["status_label"],
        "model_name": pred["model_name"],
        "shap_features": shap_features,
        "risk_profile": profile,
        "risk_reason": reason,
        "recommendations": recommendations,
        "add_on_count": addon_count,
        "customer_raw": customer_raw,
    }


def report_to_summary_row(report: dict, customer_id: Optional[str] = None) -> dict:
    """
    Ekstrak ringkasan satu-baris dari hasil build_full_report() — dipakai untuk
    tabel hasil bulk prediction dan ekspor CSV.
    """
    raw = report["customer_raw"]
    top_shap_str = " | ".join(f["readable"] for f in report["shap_features"]) if report["shap_features"] else ""
    top_rec = report["recommendations"][0] if report["recommendations"] else ""

    return {
        "CustomerID": customer_id or raw.get("customerID", ""),
        "Segment": report["segment"],
        "Tenure": raw.get("tenure"),
        "MonthlyCharges": raw.get("MonthlyCharges"),
        "Contract": raw.get("Contract"),
        "ChurnProbability": round(report["churn_probability"], 4),
        "ChurnScore": round(report["churn_score"], 1),
        "RiskZone": report["risk_zone"],
        "Status": report["status_label"],
        "RiskProfile": report["risk_profile"],
        "TopChurnDrivers": top_shap_str,
        "TopRecommendation": top_rec,
    }


# Schema input untuk form & bulk validation
REQUIRED_COLUMNS_FOR_PREDICTION = list(MODEL_FEATURE_COLUMNS)
