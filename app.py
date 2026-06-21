"""
FINTel Customer Churn Intelligence Dashboard
==============================================
Aplikasi Streamlit untuk tim retention/customer success: cari satu customer
berdasarkan Customer ID, lihat laporan churn lengkap (skor, profil risiko,
SHAP drivers, dan rekomendasi aksi) — setara dengan output fungsi
`predict_and_recommend()` pada notebook riset.

Jalankan: streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.prediction import (
    predict_customer, find_customer, compute_add_on_count,
    GRAY_ZONE_LOWER, GRAY_ZONE_UPPER,
)
from utils.shap_utils import (
    compute_shap_for_customer, get_top_shap_features, format_shap_features_for_display,
)
from utils.recommendation import assign_churn_profile, generate_recommendation, DEFAULT_MEDIAN_MONTHLY

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
    with st.sidebar:
        st.markdown("### 🔍 Cari Customer")
        customer_id_input = st.text_input(
            "Customer ID",
            placeholder="contoh: 7590-VHVEG",
            help="Masukkan Customer ID persis seperti pada sistem (tidak case-sensitive).",
        )
        search_clicked = st.button("Cari Customer", type="primary", use_container_width=True)

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

    return customer_id_input, search_clicked


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
    median_monthly = metadata.get("median_monthly_charges", DEFAULT_MEDIAN_MONTHLY)

    customer_id_input, search_clicked = render_sidebar(metadata)

    if not customer_id_input and not search_clicked:
        st.info(
            "👋 Selamat datang! Masukkan **Customer ID** di sidebar kiri untuk melihat "
            "laporan churn lengkap pelanggan — skor risiko, profil, faktor pendorong (SHAP), "
            "dan rekomendasi aksi retensi."
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

    # ── Jalankan prediksi end-to-end ────────────────────────────────────
    with st.spinner("Menghitung prediksi churn..."):
        cold_bundle = artifacts["cold_bundle"]
        noncold_bundle = artifacts["noncold_bundle"]
        cold_shap = artifacts["cold_shap"]
        noncold_shap = artifacts["noncold_shap"]

        # add_on_count untuk risk-profiling (dihitung langsung dari data mentah)
        addon_count = customer_row.get("add_on_count")
        if pd.isna(addon_count) if addon_count is not None else True:
            addon_count = compute_add_on_count(customer_row)

        pred = predict_customer(customer_row, cold_bundle, noncold_bundle)

        model_pipeline = cold_bundle["model"] if pred["is_cold_start"] else noncold_bundle["model"]
        shap_bundle = cold_shap if pred["is_cold_start"] else noncold_shap

        try:
            shap_row, feature_names = compute_shap_for_customer(model_pipeline, shap_bundle, pred["X_input"])
            top_shap = get_top_shap_features(shap_row, feature_names, top_n=3)
            shap_display = format_shap_features_for_display(top_shap)
        except Exception:
            shap_display = []
            top_shap = []

        customer_raw = customer_row.to_dict()
        customer_raw["add_on_count"] = addon_count

        profile, reason = assign_churn_profile(
            customer_raw, is_cold_start=pred["is_cold_start"], median_monthly=median_monthly
        )
        recommendations = generate_recommendation(profile, risk_zone=pred["risk_zone"])

    # ── Render laporan ───────────────────────────────────────────────────
    segment_label = "Cold Start" if pred["is_cold_start"] else "Non-Cold Start"

    render_customer_overview(customer_row, segment_label)
    render_prediction_summary(
        pred["churn_score"], pred["churn_probability"], pred["risk_zone"], pred["status_label"]
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_risk_profile(profile, reason)
    with col_right:
        render_top_shap_drivers(shap_display)

    render_recommendations(recommendations, pred["risk_zone"])
    render_customer_details(customer_row)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Model: `{pred['model_name']}` · Threshold: `{pred['threshold']:.4f}` · "
        f"Segment: `{segment_label}` · FINTel Churn Intelligence © 2026"
    )


if __name__ == "__main__":
    main()
