"""
FINTel Customer Churn Intelligence Dashboard
==============================================
Aplikasi Streamlit untuk tim Retention / Customer Success & C-level
stakeholder. Tiga mode operasi:

1. Existing Customer  — cari customer terdaftar berdasarkan Customer ID
2. New Customer       — input manual untuk skenario / prospek baru
3. Bulk Prediction    — upload CSV/Excel berisi banyak customer sekaligus

Jalankan: streamlit run app.py
"""

import io
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.prediction import (
    find_customer, GRAY_ZONE_LOWER, GRAY_ZONE_UPPER, MODEL_FEATURE_COLUMNS,
)
from utils.report_builder import (
    build_full_report, report_to_summary_row, REQUIRED_COLUMNS_FOR_PREDICTION,
)
from utils.recommendation import DEFAULT_MEDIAN_MONTHLY

# ──────────────────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FINTel Customer Churn Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ──────────────────────────────────────────────────────────────────────────
# Theming — Light mode, palet warna konsisten, kontras terjaga
# ──────────────────────────────────────────────────────────────────────────
COLOR_PRIMARY = "#1D3557"      # navy - header/text utama
COLOR_ACCENT = "#2563EB"       # biru - aksen interaktif
COLOR_SAFE = "#15803D"         # hijau - AMAN
COLOR_SAFE_BG = "#ECFDF5"
COLOR_AMBIGU = "#B45309"       # kuning tua (kontras lebih baik dari kuning murni)
COLOR_AMBIGU_BG = "#FFFBEB"
COLOR_HIGH = "#B91C1C"         # merah - TINGGI
COLOR_HIGH_BG = "#FEF2F2"
COLOR_TEXT = "#1F2937"
COLOR_TEXT_MUTED = "#4B5563"
COLOR_BORDER = "#E5E7EB"
COLOR_CARD_BG = "#FFFFFF"
COLOR_PAGE_BG = "#FFFFFF"

RISK_ZONE_STYLE = {
    "AMAN": {"color": COLOR_SAFE, "bg": COLOR_SAFE_BG, "label": "AMAN"},
    "AMBIGU": {"color": COLOR_AMBIGU, "bg": COLOR_AMBIGU_BG, "label": "AMBIGU"},
    "TINGGI": {"color": COLOR_HIGH, "bg": COLOR_HIGH_BG, "label": "TINGGI"},
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {COLOR_PAGE_BG};
    }}
    /* Hapus padding besar default agar dashboard terasa lebih rapat & profesional */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}
    h1, h2, h3, h4 {{
        color: {COLOR_PRIMARY} !important;
        font-weight: 700 !important;
    }}
    p, span, label, div {{
        color: {COLOR_TEXT};
    }}
    /* Header utama */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 0 0 0;
    }}
    .app-header .logo-badge {{
        width: 52px;
        height: 52px;
        border-radius: 12px;
        background: linear-gradient(135deg, {COLOR_ACCENT}, {COLOR_PRIMARY});
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        flex-shrink: 0;
    }}
    .app-header .title-block h1 {{
        margin: 0;
        font-size: 1.65rem;
        line-height: 1.2;
    }}
    .app-header .title-block p {{
        margin: 2px 0 0 0;
        color: {COLOR_TEXT_MUTED};
        font-size: 0.95rem;
        font-weight: 500;
    }}
    hr.fintel-divider {{
        border: none;
        border-top: 1px solid {COLOR_BORDER};
        margin: 18px 0 22px 0;
    }}
    /* Generic card */
    .fintel-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .fintel-card h3 {{
        margin-top: 0;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .fintel-section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        margin: 6px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    /* Overview grid */
    .overview-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }}
    .overview-item {{
        background: #F9FAFB;
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 12px 14px;
    }}
    .overview-item .ov-label {{
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {COLOR_TEXT_MUTED};
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .overview-item .ov-value {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
    }}
    /* Metric cards (Prediction Summary) */
    .metric-card {{
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid {COLOR_BORDER};
        height: 100%;
    }}
    .metric-card .m-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
        opacity: 0.85;
        margin-bottom: 6px;
    }}
    .metric-card .m-value {{
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-card .m-sub {{
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
        opacity: 0.9;
    }}
    /* Risk profile box */
    .risk-profile-box {{
        background: #F5F7FB;
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 10px;
        padding: 16px 20px;
    }}
    .risk-profile-box .rp-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {COLOR_PRIMARY};
        margin-bottom: 6px;
    }}
    .risk-profile-box .rp-reason {{
        color: {COLOR_TEXT_MUTED};
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    /* Recommendation cards */
    .rec-card {{
        background: #F9FAFB;
        border: 1px solid {COLOR_BORDER};
        border-left: 4px solid {COLOR_ACCENT};
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: {COLOR_TEXT};
        display: flex;
        gap: 10px;
    }}
    .rec-card .rec-num {{
        font-weight: 800;
        color: {COLOR_ACCENT};
        flex-shrink: 0;
    }}
    .rec-card.priority {{
        border-left-color: {COLOR_HIGH};
        background: {COLOR_HIGH_BG};
    }}
    .rec-card.priority .rec-num {{
        color: {COLOR_HIGH};
    }}
    /* Gray zone warning banner */
    .gray-zone-banner {{
        background: {COLOR_AMBIGU_BG};
        border: 1px solid #FCD34D;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
        color: #92400E;
        font-size: 0.93rem;
        line-height: 1.5;
    }}
    .gray-zone-banner b {{
        color: #92400E;
    }}
    .high-risk-banner {{
        background: {COLOR_HIGH_BG};
        border: 1px solid #FECACA;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
        color: #7F1D1D;
        font-size: 0.93rem;
        line-height: 1.5;
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #F9FAFB;
        border-right: 1px solid {COLOR_BORDER};
    }}
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {COLOR_PRIMARY} !important;
    }}
    .sidebar-zone-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.85rem;
    }}
    .sidebar-zone-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    /* Badge for segment */
    .segment-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: #EFF6FF;
        color: {COLOR_ACCENT};
        border: 1px solid #BFDBFE;
    }}
    @media (max-width: 900px) {{
        .overview-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Data & Model Loading (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_bundle(filename: str) -> dict:
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_shap_bundle(filename: str) -> dict:
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    path = DATA_DIR / "customer_data.csv"
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = MODELS_DIR / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_all_artifacts():
    errors = []
    cold_bundle = noncold_bundle = cold_shap = noncold_shap = None
    customer_df = None
    metadata = {}

    try:
        cold_bundle = load_model_bundle("cold_start_model.pkl")
    except Exception as e:
        errors.append(f"Gagal memuat cold_start_model.pkl: {e}")
    try:
        noncold_bundle = load_model_bundle("noncold_model.pkl")
    except Exception as e:
        errors.append(f"Gagal memuat noncold_model.pkl: {e}")
    try:
        cold_shap = load_shap_bundle("cold_shap.pkl")
    except Exception as e:
        errors.append(f"Gagal memuat cold_shap.pkl: {e}")
    try:
        noncold_shap = load_shap_bundle("noncold_shap.pkl")
    except Exception as e:
        errors.append(f"Gagal memuat noncold_shap.pkl: {e}")
    try:
        customer_df = load_customer_data()
    except Exception as e:
        errors.append(f"Gagal memuat data/customer_data.csv: {e}")
    try:
        metadata = load_metadata()
    except Exception as e:
        errors.append(f"Gagal memuat model_metadata.json: {e}")

    return {
        "cold_bundle": cold_bundle,
        "noncold_bundle": noncold_bundle,
        "cold_shap": cold_shap,
        "noncold_shap": noncold_shap,
        "customer_df": customer_df,
        "metadata": metadata,
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────────
# UI Helper Components
# ──────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <div class="logo-badge">📡</div>
            <div class="title-block">
                <h1>FINTel Customer Churn Intelligence Dashboard</h1>
                <p>AI-Powered Customer Retention System</p>
            </div>
        </div>
        <hr class="fintel-divider">
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: dict):
    """Render sidebar dengan mode selector + info model. Return mode terpilih."""
    with st.sidebar:
        st.markdown("### 🎯 Mode Prediksi")
        mode = st.radio(
            "Pilih mode:",
            options=["Existing Customer", "New Customer", "Bulk Prediction"],
            captions=[
                "Cari pelanggan yang sudah terdaftar",
                "Input manual untuk prospek/skenario baru",
                "Upload CSV/Excel banyak pelanggan",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ℹ️ About Model")

        cold_meta = metadata.get("cold_start", {})
        noncold_meta = metadata.get("non_cold_start", {})

        st.markdown(
            f"""
            **Pendekatan:** Dual-pipeline segmentation

            Model dilatih terpisah untuk dua segmen pelanggan agar setiap
            model mempelajari pola churn yang spesifik:

            **🟦 Cold Start** *(tenure ≤ {metadata.get('cold_start_tenure_cutoff', 5)} bulan)*
            - Model: `{cold_meta.get('model_name', 'N/A')}`
            - Threshold: `{cold_meta.get('threshold', 0):.4f}`
            - Recall: `{cold_meta.get('metrics', {}).get('recall', 0):.1%}`
            - ROC-AUC: `{cold_meta.get('metrics', {}).get('roc_auc', 0):.3f}`

            **🟩 Non-Cold Start** *(tenure > {metadata.get('cold_start_tenure_cutoff', 5)} bulan)*
            - Model: `{noncold_meta.get('model_name', 'N/A')}`
            - Threshold: `{noncold_meta.get('threshold', 0):.4f}`
            - Recall: `{noncold_meta.get('metrics', {}).get('recall', 0):.1%}`
            - ROC-AUC: `{noncold_meta.get('metrics', {}).get('roc_auc', 0):.3f}`

            Interpretasi (SHAP): `{cold_meta.get('shap_explainer', 'LinearExplainer')}`
            """
        )

        st.markdown("---")
        st.markdown("### 🚦 Risk Zone")
        gray_lower = metadata.get("gray_zone_lower", GRAY_ZONE_LOWER)
        gray_upper = metadata.get("gray_zone_upper", GRAY_ZONE_UPPER)
        st.markdown(
            f"""
            <div class="sidebar-zone-row">
                <div class="sidebar-zone-dot" style="background:{COLOR_SAFE}"></div>
                <div><b>AMAN</b> — Churn Score &lt; {gray_lower}</div>
            </div>
            <div class="sidebar-zone-row">
                <div class="sidebar-zone-dot" style="background:{COLOR_AMBIGU}"></div>
                <div><b>AMBIGU</b> — {gray_lower} ≤ Churn Score ≤ {gray_upper}</div>
            </div>
            <div class="sidebar-zone-row">
                <div class="sidebar-zone-dot" style="background:{COLOR_HIGH}"></div>
                <div><b>TINGGI</b> — Churn Score &gt; {gray_upper}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Zona AMBIGU menandakan pelanggan masih bimbang antara churn atau "
            "bertahan — prioritas tindak lanjut cepat (24-48 jam)."
        )

    return mode


def render_customer_overview(customer_row: pd.Series, segment_label: str):
    st.markdown('<div class="fintel-section-title">👤 Customer Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fintel-card">
            <div class="overview-grid">
                <div class="overview-item">
                    <div class="ov-label">Customer ID</div>
                    <div class="ov-value">{customer_row.get('customerID', 'N/A')}</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Segment</div>
                    <div class="ov-value"><span class="segment-badge">{segment_label}</span></div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Tenure</div>
                    <div class="ov-value">{customer_row.get('tenure', 'N/A'):.0f} bulan</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Contract Type</div>
                    <div class="ov-value">{customer_row.get('Contract', 'N/A')}</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Internet Service</div>
                    <div class="ov-value">{customer_row.get('InternetService', 'N/A')}</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Monthly Charges</div>
                    <div class="ov-value">${customer_row.get('MonthlyCharges', 0):,.2f}</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Total Charges</div>
                    <div class="ov-value">${customer_row.get('TotalCharges', 0):,.2f}</div>
                </div>
                <div class="overview-item">
                    <div class="ov-label">Payment Method</div>
                    <div class="ov-value">{customer_row.get('PaymentMethod', 'N/A')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary(churn_score: float, churn_prob: float, risk_zone: str, status_label: str):
    st.markdown('<div class="fintel-section-title">📊 Prediction Summary</div>', unsafe_allow_html=True)

    zone_style = RISK_ZONE_STYLE[risk_zone]
    status_color = COLOR_HIGH if status_label == "BERISIKO CHURN" else COLOR_SAFE
    status_bg = COLOR_HIGH_BG if status_label == "BERISIKO CHURN" else COLOR_SAFE_BG

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{zone_style['bg']}; border-color:{zone_style['color']}33;">
                <div class="m-label" style="color:{zone_style['color']}">Churn Score</div>
                <div class="m-value" style="color:{zone_style['color']}">{churn_score:.0f}<span style="font-size:1rem;">/100</span></div>
                <div class="m-sub" style="color:{zone_style['color']}">Titik kritis = 54</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card" style="background:#F5F7FB; border-color:{COLOR_BORDER};">
                <div class="m-label" style="color:{COLOR_PRIMARY}">Churn Probability</div>
                <div class="m-value" style="color:{COLOR_PRIMARY}">{churn_prob:.1%}</div>
                <div class="m-sub" style="color:{COLOR_TEXT_MUTED}">Probabilitas model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{zone_style['bg']}; border-color:{zone_style['color']}33;">
                <div class="m-label" style="color:{zone_style['color']}">Risk Zone</div>
                <div class="m-value" style="color:{zone_style['color']}">{zone_style['label']}</div>
                <div class="m-sub" style="color:{zone_style['color']}">Berdasarkan Churn Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{status_bg}; border-color:{status_color}33;">
                <div class="m-label" style="color:{status_color}">Prediction Status</div>
                <div class="m-value" style="color:{status_color}; font-size:1.4rem;">{status_label}</div>
                <div class="m-sub" style="color:{status_color}">Hasil klasifikasi model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if risk_zone == "AMBIGU":
        st.markdown(
            f"""
            <div class="gray-zone-banner">
                ⚠️ <b>WARNING — ZONA AMBIGU (Gray Area)</b><br>
                Churn Score berada di rentang {GRAY_ZONE_LOWER}-{GRAY_ZONE_UPPER}. Pelanggan ini masih
                <b>BIMBANG</b> antara churn atau bertahan — segera tangani dengan prioritas tinggi
                sebelum keputusan jatuh ke churn.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif risk_zone == "TINGGI":
        st.markdown(
            f"""
            <div class="high-risk-banner">
                🔴 <b>RISIKO TINGGI</b> — Churn Score &gt; {GRAY_ZONE_UPPER}, pelanggan hampir pasti churn.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_profile(profile: str, reason: str):
    st.markdown('<div class="fintel-section-title">🧭 Risk Profile</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-profile-box">
            <div class="rp-title">{profile}</div>
            <div class="rp-reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_shap_drivers(shap_features: list):
    st.markdown('<div class="fintel-section-title">📈 Top Churn Drivers (SHAP)</div>', unsafe_allow_html=True)

    if not shap_features:
        st.info("SHAP belum dapat dihitung untuk customer ini.")
        return

    df_shap = pd.DataFrame(shap_features)
    df_shap = df_shap.sort_values("impact", key=lambda s: s.abs())

    colors = [COLOR_HIGH if v > 0 else COLOR_SAFE for v in df_shap["impact"]]

    fig = go.Figure(
        go.Bar(
            x=df_shap["impact"],
            y=df_shap["readable"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in df_shap["impact"]],
            textposition="outside",
            hovertemplate="%{y}<br>SHAP value: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=COLOR_TEXT, size=13),
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        xaxis=dict(title="SHAP value (dampak terhadap prediksi churn)", zeroline=True,
                    zerolinecolor=COLOR_BORDER, gridcolor="#F3F4F6"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("🔴 Mendorong churn &nbsp;&nbsp;|&nbsp;&nbsp; 🟢 Menekan churn")


def render_recommendations(recommendations: list, risk_zone: str):
    st.markdown('<div class="fintel-section-title">✅ Recommended Actions</div>', unsafe_allow_html=True)
    n_base = len(recommendations) - (1 if risk_zone == "AMBIGU" else 0)
    for i, rec in enumerate(recommendations, 1):
        is_priority = risk_zone == "AMBIGU" and i > n_base
        css_class = "rec-card priority" if is_priority else "rec-card"
        st.markdown(
            f"""<div class="{css_class}"><span class="rec-num">{i}.</span><span>{rec}</span></div>""",
            unsafe_allow_html=True,
        )


def render_customer_details(customer_row: pd.Series):
    st.markdown('<div class="fintel-section-title">📋 Customer Details</div>', unsafe_allow_html=True)
    display_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    available_cols = [c for c in display_cols if c in customer_row.index]
    detail_df = pd.DataFrame({
        "Atribut": available_cols,
        "Nilai": [customer_row.get(c) for c in available_cols],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=420)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 1: Existing Customer
# ──────────────────────────────────────────────────────────────────────────
def page_existing_customer(artifacts: dict, metadata: dict):
    customer_df = artifacts["customer_df"]
    median_monthly = metadata.get("median_monthly_charges", DEFAULT_MEDIAN_MONTHLY)

    st.markdown(
        '<div class="fintel-section-title">🔍 Cari Pelanggan Terdaftar</div>',
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        customer_id_input = st.text_input(
            "Customer ID",
            placeholder="contoh: 7590-VHVEG",
            label_visibility="collapsed",
            help="Masukkan Customer ID persis seperti pada sistem (tidak case-sensitive).",
        )
    with col_btn:
        search_clicked = st.button("Cari", type="primary", use_container_width=True)

    if not customer_id_input and not search_clicked:
        st.info(
            "👋 Masukkan **Customer ID** di atas untuk melihat laporan churn lengkap — "
            "skor risiko, profil pelanggan, faktor pendorong (SHAP), dan rekomendasi aksi retensi."
        )
        with st.expander("📌 Contoh Customer ID yang tersedia"):
            st.dataframe(
                customer_df[["customerID", "tenure", "Contract", "MonthlyCharges"]].head(10),
                use_container_width=True, hide_index=True,
            )
        return

    if not customer_id_input:
        st.warning("Silakan masukkan Customer ID terlebih dahulu.")
        return

    customer_row = find_customer(customer_id_input, customer_df)

    if customer_row is None:
        st.warning(f"⚠️ Customer ID **'{customer_id_input}'** tidak ditemukan.")
        st.caption("Periksa kembali ejaan Customer ID, atau lihat contoh ID yang tersedia di bawah.")
        with st.expander("📌 Contoh Customer ID yang tersedia"):
            st.dataframe(
                customer_df[["customerID", "tenure", "Contract", "MonthlyCharges"]].head(10),
                use_container_width=True, hide_index=True,
            )
        return

    with st.spinner("Menghitung prediksi churn..."):
        report = build_full_report(
            customer_row,
            artifacts["cold_bundle"], artifacts["noncold_bundle"],
            artifacts["cold_shap"], artifacts["noncold_shap"],
            median_monthly=median_monthly,
        )

    render_full_customer_report(customer_row, report)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2: New Customer (manual input)
# ──────────────────────────────────────────────────────────────────────────
# Dropdown options - persis sesuai unique values di data training
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def page_new_customer(artifacts: dict, metadata: dict):
    median_monthly = metadata.get("median_monthly_charges", DEFAULT_MEDIAN_MONTHLY)

    st.markdown(
        '<div class="fintel-section-title">📝 Input Data Pelanggan Baru</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Gunakan form ini untuk menilai risiko churn calon pelanggan / skenario hipotesis "
        "yang **belum terdaftar** di sistem. Semua field wajib diisi karena dipakai langsung "
        "oleh model."
    )

    with st.form("new_customer_form", clear_on_submit=False):
        # Demografi
        st.markdown("##### 👤 Demografi")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
        senior = c2.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["SeniorCitizen"])
        partner = c3.selectbox("Memiliki Pasangan", CATEGORICAL_OPTIONS["Partner"])
        dependents = c4.selectbox("Memiliki Tanggungan", CATEGORICAL_OPTIONS["Dependents"])

        # Account / Tagihan
        st.markdown("##### 💳 Account & Tagihan")
        a1, a2, a3 = st.columns(3)
        tenure = a1.number_input(
            "Tenure (bulan)", min_value=0, max_value=120, value=12, step=1,
            help="Lama berlangganan dalam bulan. Tenure ≤ 5 = segmen Cold Start."
        )
        monthly_charges = a2.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=500.0,
            value=70.0, step=0.5, format="%.2f",
        )
        total_charges = a3.number_input(
            "Total Charges ($)", min_value=0.0, value=float(tenure) * 70.0,
            step=1.0, format="%.2f",
            help="Total tagihan akumulasi sejak pelanggan mulai berlangganan."
        )

        b1, b2 = st.columns(2)
        contract = b1.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
        paperless = b2.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])

        payment = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])

        cltv = st.number_input(
            "CLTV (Customer Lifetime Value)", min_value=0, max_value=10000,
            value=4400, step=100,
            help="Estimasi nilai pelanggan bagi perusahaan (range pada data training: ~2.000–6.500)."
        )

        # Layanan
        st.markdown("##### 📞 Layanan Telepon")
        p1, p2 = st.columns(2)
        phone_service = p1.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
        multiple_lines = p2.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])

        st.markdown("##### 🌐 Layanan Internet & Add-on")
        i1, i2 = st.columns(2)
        internet_service = i1.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
        online_security = i2.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])

        i3, i4 = st.columns(2)
        online_backup = i3.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
        device_protection = i4.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])

        i5, i6 = st.columns(2)
        tech_support = i5.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
        streaming_tv = i6.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])

        i7, _ = st.columns(2)
        streaming_movies = i7.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])

        submitted = st.form_submit_button("🔮 Prediksi Churn", type="primary", use_container_width=True)

    if not submitted:
        st.info("ℹ️ Isi form di atas lalu klik **Prediksi Churn** untuk melihat laporan.")
        return

    # Konsistensi internal sederhana: kalau Phone Service = No → MultipleLines harus 'No phone service'
    if phone_service == "No" and multiple_lines != "No phone service":
        st.warning(
            "⚠️ Inkonsistensi terdeteksi: **Phone Service = No** mengharuskan **Multiple Lines = "
            "'No phone service'**. Nilai di-override otomatis."
        )
        multiple_lines = "No phone service"

    # Konsistensi: kalau Internet Service = No → semua add-on internet harus 'No internet service'
    if internet_service == "No":
        internet_addons = {
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        }
        mismatches = [k for k, v in internet_addons.items() if v != "No internet service"]
        if mismatches:
            st.warning(
                f"⚠️ Inkonsistensi terdeteksi: **Internet Service = No** mengharuskan semua layanan "
                f"internet bernilai **'No internet service'**. Field berikut di-override otomatis: "
                f"{', '.join(mismatches)}."
            )
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"

    customer_row = pd.Series({
        "customerID": "NEW-PROSPECT",
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges), "TotalCharges": float(total_charges),
        "CLTV": float(cltv),
    })

    with st.spinner("Menghitung prediksi churn..."):
        report = build_full_report(
            customer_row,
            artifacts["cold_bundle"], artifacts["noncold_bundle"],
            artifacts["cold_shap"], artifacts["noncold_shap"],
            median_monthly=median_monthly,
        )

    st.success("✅ Prediksi berhasil dijalankan untuk pelanggan baru.")
    render_full_customer_report(customer_row, report)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3: Bulk Prediction
# ──────────────────────────────────────────────────────────────────────────
def _make_bulk_template() -> bytes:
    """Buat CSV template berisi 2 contoh baris + header lengkap."""
    template_rows = [
        {
            "customerID": "PROSPECT-001",
            "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.50, "TotalCharges": 179.00, "CLTV": 3500,
        },
        {
            "customerID": "PROSPECT-002",
            "gender": "Male", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
            "tenure": 36, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
            "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.20, "TotalCharges": 2347.20, "CLTV": 5200,
        },
    ]
    df_template = pd.DataFrame(template_rows)
    return df_template.to_csv(index=False).encode("utf-8")


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Baca CSV / Excel berdasarkan ekstensi file."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan .csv, .xlsx, atau .xls.")


def _validate_bulk_input(df: pd.DataFrame) -> tuple:
    """
    Validasi bahwa df berisi semua kolom yang dibutuhkan model + opsional customerID.
    Returns (is_valid, error_message, df_normalized).
    """
    required = set(REQUIRED_COLUMNS_FOR_PREDICTION)
    df_cols = set(df.columns)

    missing = required - df_cols
    if missing:
        return False, f"Kolom wajib tidak ditemukan: {sorted(missing)}", None

    if "customerID" not in df.columns:
        df = df.copy()
        df.insert(0, "customerID", [f"BULK-{i+1:04d}" for i in range(len(df))])

    if df.empty:
        return False, "File terupload kosong (tidak ada baris data).", None

    return True, None, df


def page_bulk_prediction(artifacts: dict, metadata: dict):
    median_monthly = metadata.get("median_monthly_charges", DEFAULT_MEDIAN_MONTHLY)

    st.markdown(
        '<div class="fintel-section-title">📦 Bulk Prediction (CSV / Excel)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload file berisi banyak pelanggan sekaligus untuk diprediksi serentak. "
        "Cocok untuk skenario evaluasi prospek campaign / pipeline sales."
    )

    with st.expander("📋 Format file yang dibutuhkan", expanded=False):
        st.markdown(
            "File CSV/Excel wajib berisi **kolom-kolom berikut** (urutan bebas):"
        )
        cols_display = ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS_FOR_PREDICTION)
        st.markdown(cols_display)
        st.markdown(
            "Kolom `customerID` **opsional** — jika tidak ada, aplikasi akan "
            "men-generate ID otomatis (BULK-0001, BULK-0002, ...). Unduh template "
            "di bawah untuk memastikan format yang benar:"
        )
        st.download_button(
            "⬇️ Download Template CSV",
            data=_make_bulk_template(),
            file_name="fintel_bulk_template.csv",
            mime="text/csv",
            use_container_width=False,
        )

    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        help="Maksimal 200 MB. Untuk dataset sangat besar (>10.000 baris), pertimbangkan batch lebih kecil.",
    )

    if uploaded_file is None:
        st.info("📤 Upload file di atas untuk memulai bulk prediction.")
        return

    try:
        df_input = _read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        return

    st.caption(f"📄 File **{uploaded_file.name}** dimuat: {df_input.shape[0]} baris, {df_input.shape[1]} kolom.")

    is_valid, err, df_norm = _validate_bulk_input(df_input)
    if not is_valid:
        st.error(f"❌ Validasi gagal: {err}")
        with st.expander("🔍 Lihat kolom file yang terupload"):
            st.write(list(df_input.columns))
        return

    n_rows = len(df_norm)
    st.write(f"✅ Validasi lolos. Memproses **{n_rows}** pelanggan...")

    cold_bundle = artifacts["cold_bundle"]
    noncold_bundle = artifacts["noncold_bundle"]
    cold_shap = artifacts["cold_shap"]
    noncold_shap = artifacts["noncold_shap"]

    progress = st.progress(0.0, text="Memulai prediksi...")
    results = []
    errors = []

    for i, (_, row) in enumerate(df_norm.iterrows()):
        try:
            report = build_full_report(
                row, cold_bundle, noncold_bundle, cold_shap, noncold_shap,
                median_monthly=median_monthly,
            )
            results.append(report_to_summary_row(report, customer_id=row.get("customerID")))
        except Exception as e:
            errors.append({"customerID": row.get("customerID", f"ROW-{i+1}"), "error": str(e)})

        # Update progress bar setiap 1% atau setiap baris kalau dataset kecil
        if n_rows <= 100 or (i + 1) % max(1, n_rows // 100) == 0 or (i + 1) == n_rows:
            progress.progress((i + 1) / n_rows, text=f"Memproses {i+1}/{n_rows} pelanggan...")

    progress.empty()

    if errors:
        st.warning(f"⚠️ {len(errors)} pelanggan gagal diprediksi. Lihat detail di bawah.")
        with st.expander("Lihat error detail"):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    if not results:
        st.error("❌ Tidak ada pelanggan yang berhasil diprediksi. Periksa format/nilai data.")
        return

    df_results = pd.DataFrame(results)

    # ── Ringkasan agregat ──────────────────────────────────────────────
    n_success = len(df_results)
    n_churn = int(df_results["Status"].eq("BERISIKO CHURN").sum())
    n_high = int(df_results["RiskZone"].eq("TINGGI").sum())
    n_ambigu = int(df_results["RiskZone"].eq("AMBIGU").sum())
    n_aman = int(df_results["RiskZone"].eq("AMAN").sum())

    st.markdown('<div class="fintel-section-title">📊 Ringkasan Hasil</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Diprediksi", f"{n_success:,}")
    s2.metric("Berisiko Churn", f"{n_churn:,}", delta=f"{n_churn/n_success:.0%}")
    s3.metric("Risk: TINGGI", f"{n_high:,}", delta=f"{n_high/n_success:.0%}", delta_color="inverse")
    s4.metric("Risk: AMBIGU", f"{n_ambigu:,}", delta=f"{n_ambigu/n_success:.0%}", delta_color="off")
    s5.metric("Risk: AMAN", f"{n_aman:,}", delta=f"{n_aman/n_success:.0%}")

    # ── Filter / tabel detail ──────────────────────────────────────────
    st.markdown('<div class="fintel-section-title">📋 Detail Hasil per Pelanggan</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([1, 1])
    risk_filter = f1.multiselect(
        "Filter Risk Zone",
        options=["TINGGI", "AMBIGU", "AMAN"],
        default=["TINGGI", "AMBIGU", "AMAN"],
    )
    profile_filter = f2.multiselect(
        "Filter Risk Profile",
        options=sorted(df_results["RiskProfile"].unique().tolist()),
        default=sorted(df_results["RiskProfile"].unique().tolist()),
    )

    df_filtered = df_results[
        df_results["RiskZone"].isin(risk_filter) & df_results["RiskProfile"].isin(profile_filter)
    ].sort_values("ChurnScore", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "ChurnScore": st.column_config.ProgressColumn(
                "Churn Score", format="%.0f", min_value=0, max_value=100,
            ),
            "ChurnProbability": st.column_config.NumberColumn(
                "Probability", format="%.2f",
            ),
        },
    )

    # ── Download hasil ─────────────────────────────────────────────────
    st.markdown('<div class="fintel-section-title">⬇️ Download Hasil</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
    d1.download_button(
        "Download CSV (semua hasil)", data=csv_bytes,
        file_name=f"fintel_bulk_prediction_{timestamp}.csv",
        mime="text/csv", use_container_width=True,
    )

    # Excel download
    excel_buf = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Predictions", index=False)
        d2.download_button(
            "Download Excel (semua hasil)", data=excel_buf.getvalue(),
            file_name=f"fintel_bulk_prediction_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except ImportError:
        d2.info("Excel export memerlukan `openpyxl`. Gunakan CSV download di sebelah kiri.")


# ──────────────────────────────────────────────────────────────────────────
# Render report (shared antara Existing & New Customer)
# ──────────────────────────────────────────────────────────────────────────
def render_full_customer_report(customer_row, report: dict):
    """Render seluruh komponen laporan untuk satu pelanggan."""
    if isinstance(customer_row, dict):
        customer_row = pd.Series(customer_row)

    segment_label = "Cold Start" if report["is_cold_start"] else "Non-Cold Start"

    render_customer_overview(customer_row, segment_label)
    render_prediction_summary(
        report["churn_score"], report["churn_probability"],
        report["risk_zone"], report["status_label"],
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_risk_profile(report["risk_profile"], report["risk_reason"])
    with col_right:
        render_top_shap_drivers(report["shap_features"])

    render_recommendations(report["recommendations"], report["risk_zone"])
    render_customer_details(customer_row)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Model: `{report['model_name']}` · Threshold: `{report['threshold']:.4f}` · "
        f"Segment: `{segment_label}` · FINTel Churn Intelligence © 2026"
    )


# ──────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────
def main():
    render_header()

    artifacts = load_all_artifacts()

    if artifacts["errors"]:
        st.error(
            "**Gagal memuat model/data.** Pastikan folder `models/` dan `data/` sudah terisi "
            "(jalankan `train_model.py` terlebih dahulu, atau cek struktur folder)."
        )
        for err in artifacts["errors"]:
            st.code(err)
        st.stop()

    customer_df = artifacts["customer_df"]
    if customer_df is None or customer_df.empty:
        st.error("Data kosong: `data/customer_data.csv` tidak berisi data pelanggan.")
        st.stop()

    metadata = artifacts["metadata"]
    mode = render_sidebar(metadata)

    if mode == "Existing Customer":
        page_existing_customer(artifacts, metadata)
    elif mode == "New Customer":
        page_new_customer(artifacts, metadata)
    elif mode == "Bulk Prediction":
        page_bulk_prediction(artifacts, metadata)


if __name__ == "__main__":
    main()
