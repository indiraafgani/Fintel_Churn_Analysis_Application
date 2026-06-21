"""
recommendation.py
==================
Churn Risk Profile Matcher + Recommendation Engine.

Mereplikasi PERSIS logika `assign_churn_profile()` dan `recommendation_map`
dari notebook (sel "Churn Risk Profile Matcher").
"""

# Statistik referensi dari dataset asli (median MonthlyCharges & add_on_count).
# Nilai default di bawah adalah fallback; nilai aktual sebaiknya dihitung dari
# data customer_data.csv saat aplikasi start (lihat app.py: load_reference_stats).
DEFAULT_MEDIAN_MONTHLY = 70.35
DEFAULT_MEDIAN_ADDON = 2.0


def assign_churn_profile(customer_raw: dict, is_cold_start: bool = False,
                          median_monthly: float = DEFAULT_MEDIAN_MONTHLY) -> tuple:
    """
    Assign Churn Risk Profile berdasarkan data customer mentah.

    Parameters
    ----------
    customer_raw : dict
        Data asli customer (sebelum encoding), minimal berisi:
        tenure, MonthlyCharges, Contract, add_on_count
    is_cold_start : bool
        True jika customer berasal dari segmen Cold Start (tenure <= 5 bulan)
    median_monthly : float
        Median MonthlyCharges dari basis pelanggan (dipakai sebagai ambang
        "Price Sensitive")

    Returns
    -------
    (profile: str, reason: str)
    """
    tenure_val = customer_raw.get('tenure', 0) or 0
    monthly_val = customer_raw.get('MonthlyCharges', 0) or 0
    contract_val = customer_raw.get('Contract', '') or ''
    addon_val = customer_raw.get('add_on_count', 0) or 0

    # Early Leaver — fokus cold-start dengan kontrak month-to-month
    if is_cold_start and contract_val == 'Month-to-month':
        return (
            'Early Leaver',
            'Cold Start (tenure <=5 bln) & kontrak month-to-month — risiko sangat tinggi di awal lifecycle'
        )
    if tenure_val <= 12 and contract_val == 'Month-to-month':
        return (
            'Early Leaver',
            'Tenure <=12 bulan & kontrak month-to-month — risiko tinggi di awal lifecycle'
        )
    if addon_val <= 1 and 12 < tenure_val <= 36:
        return (
            'Underserved Customer',
            'Add-on <=1, tenure menengah — customer belum menggunakan layanan secara penuh'
        )
    if monthly_val > median_monthly and addon_val <= 2:
        return (
            'Price Sensitive',
            f'MonthlyCharges >${median_monthly:.0f} dengan sedikit add-on — tidak merasakan value sepadan'
        )
    if tenure_val > 36:
        return (
            'At-Risk Loyal',
            'Tenure >36 bulan — pelanggan lama yang mulai menunjukkan sinyal churn'
        )
    return (
        'Price Sensitive',
        'Tidak cocok profil spesifik — diperlakukan sebagai Price Sensitive'
    )


# Recommendation Engine — identik dengan notebook
RECOMMENDATION_MAP = {
    'Early Leaver': [
        'Tawarkan diskon upgrade ke kontrak tahunan (hemat 15-20%)',
        'Program onboarding intensif bulan 1-3 — welcome call, tutorial produk',
        'Trial gratis 1 bulan TechSupport atau OnlineSecurity',
        'Notifikasi personal dari customer success: "Kami ingin memastikan Anda puas"',
    ],
    'Price Sensitive': [
        'Tawarkan paket bundling yang lebih hemat untuk layanan yang dimiliki',
        'Komunikasikan value proposition secara eksplisit via email/SMS',
        'Berikan loyalty points atau cashback untuk pembayaran tepat waktu',
        'Review pricing kompetitif untuk segmen ini',
    ],
    'Underserved Customer': [
        'Campaign edukasi layanan tambahan — demo produk personal',
        'Trial 1 bulan gratis add-on sesuai profil (contoh: OnlineSecurity)',
        'Personal outreach dari customer success team',
        'Newsletter fitur yang belum digunakan dengan tutorial singkat',
    ],
    'At-Risk Loyal': [
        'Loyalty reward eksklusif: upgrade layanan gratis atau diskon khusus pelanggan setia',
        'Dedicated account manager atau priority support',
        'Survei kepuasan personal — prioritas tertinggi tim retensi',
        'Undangan program beta/early access sebagai bentuk apresiasi loyalitas',
    ],
}

PRIORITY_GRAY_ZONE_ACTION = (
    '[PRIORITAS] Hubungi pelanggan dalam 24-48 jam — momentum keputusan masih terbuka'
)


def generate_recommendation(profile: str, risk_zone: str = None) -> list:
    """
    Ambil daftar rekomendasi aksi untuk profil tertentu. Jika risk_zone
    'AMBIGU', tambahkan rekomendasi prioritas kontak cepat (identik dengan
    notebook predict_and_recommend()).
    """
    recs = list(RECOMMENDATION_MAP.get(profile, ['Tidak ada rekomendasi tersedia']))
    if risk_zone == 'AMBIGU':
        recs.append(PRIORITY_GRAY_ZONE_ACTION)
    return recs


PROFILE_COLORS = {
    'Early Leaver': '#E85C5C',
    'Price Sensitive': '#F5A623',
    'Underserved Customer': '#D4A72C',
    'At-Risk Loyal': '#4C9BE8',
}
