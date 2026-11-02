import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SAYFA KONFİGÜRASYONU VE KURUMSAL STİL (CSS)
# ==============================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Yönetici Özeti | B2B Kitap Dağıtım KDS",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_executive_css():
    st.markdown(
        """
        <style>
        /* Genel Arka Plan ve Tipografi */
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        /* Üst Yönetici (Hero) Banner */
        .exec-hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%);
            color: #FFFFFF;
            padding: 24px 28px;
            border-radius: 14px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.28);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        .exec-hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
            color: #FFFFFF;
        }
        .exec-hero-subtitle {
            font-size: 0.98rem;
            color: #DBEAFE;
            margin: 0;
            font-weight: 400;
        }
        .exec-badge-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .exec-badge {
            background: rgba(255, 255, 255, 0.14);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #F8FAFC;
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        /* Özel KPI Kartları */
        .kpi-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #2563EB;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            height: 100%;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
        }
        .kpi-card.emerald { border-left-color: #10B981; }
        .kpi-card.indigo  { border-left-color: #6366F1; }
        .kpi-card.amber   { border-left-color: #F59E0B; }
        .kpi-card.rose    { border-left-color: #F43F5E; }
        .kpi-card.cyan    { border-left-color: #06B6D4; }

        .kpi-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748B;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 8px;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 0.80rem;
            color: #475569;
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .delta-up {
            background: #DCFCE7;
            color: #166534;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.75rem;
        }
        .delta-down {
            background: #FEE2E2;
            color: #991B1B;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.75rem;
        }
        .delta-neutral {
            background: #F1F5F9;
            color: #334155;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        /* Bölüm Başlıkları */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0F172A;
            margin: 18px 0 10px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid #E2E8F0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* İçgörü Kutuları */
        .insight-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #3B82F6;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .insight-box.success { border-left-color: #10B981; background: #F0FDF4; }
        .insight-box.warning { border-left-color: #F59E0B; background: #FFFBEB; }
        .insight-box.danger  { border-left-color: #EF4444; background: #FEF2F2; }
        .insight-title {
            font-weight: 700;
            font-size: 0.92rem;
            color: #0F172A;
            margin-bottom: 4px;
        }
        .insight-text {
            font-size: 0.86rem;
            color: #334155;
            margin: 0;
            line-height: 1.45;
        }
        /* Sağlık Skoru Kartı */
        .health-pill {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
        }
        .health-score {
            font-size: 1.4rem;
            font-weight: 800;
            color: #1E3A8A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. VERİ YÜKLEME, ÖN İŞLEME VE TÜRETİLMİŞ METRİKLER
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    B2B kitap dağıtım veri setini okur (Parquet öncelikli, yoksa Excel),
    temizler ve yönetici özeti için gerekli finansal/operasyonel sütunları türetir.
    """
    parquet_path = "B2B_Transaction_Data_.parquet"
    excel_path = "B2B_Transaction_Data_.xlsx"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
    else:
        st.error(
            "Veri dosyası bulunamadı! Lütfen proje dizininde "
            "'B2B_Transaction_Data_.parquet' veya 'B2B_Transaction_Data_.xlsx' dosyasının bulunduğundan emin olun."
        )
        st.stop()

    # Sütun isimlerindeki olası boşlukları temizle
    df.columns = [str(c).strip() for c in df.columns]

    # Tarih dönüşümü
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()

    # Sayısal dönüşümler
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)

    has_net_price = "NetPrice" in df.columns
    if has_net_price:
        df["NetPrice"] = pd.to_numeric(df["NetPrice"], errors="coerce").fillna(df["UnitPrice"])
    else:
        df["NetPrice"] = df["UnitPrice"]

    # İptal / İade tespiti
    df["InvoiceNo_Str"] = df["InvoiceNo"].astype(str).str.strip()
    df["IsReturn"] = (
        df["InvoiceNo_Str"].str.upper().str.startswith("C")
        | df["InvoiceNo_Str"].str.upper().str.contains("IADE")
        | (df["Quantity"] < 0)
    )

    # Brüt ve Net Ciro Hesabı
    # NetPrice birim net fiyat mı yoksa satır toplamı mı otomatik kontrol et
    valid_mask = (df["UnitPrice"] > 0) & (df["Quantity"] > 0)
    if has_net_price and valid_mask.any():
        ratio_to_unit = (df.loc[valid_mask, "NetPrice"] / df.loc[valid_mask, "UnitPrice"]).median()
        if 0.1 <= ratio_to_unit <= 1.5:
            # NetPrice birim indirimli fiyattır
            df["GrossAmount"] = df["Quantity"] * df["UnitPrice"]
            df["TotalAmount"] = df["Quantity"] * df["NetPrice"]
        else:
            # NetPrice satır tutarıysa veya farklı ölçekteyse
            df["GrossAmount"] = df["Quantity"] * df["UnitPrice"]
            df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    else:
        df["GrossAmount"] = df["Quantity"] * df["UnitPrice"]
        df["TotalAmount"] = df["GrossAmount"]

    df["DiscountAmount"] = np.maximum(0.0, df["GrossAmount"] - df["TotalAmount"])

    # Kategorik ve Metin Alanlarının Temizlenmesi
    if "City" in df.columns:
        df["City"] = df["City"].fillna("Bilinmiyor").astype(str).str.strip()
    else:
        df["City"] = "Bilinmiyor"

    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna("Genel / Diğer").astype(str).str.strip()
    else:
        df["Category"] = "Genel Kitap"

    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("Tanımsız Kitap").astype(str).str.strip()
    else:
        df["Description"] = "Tanımsız Kitap"

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna("Anonim Bayi").astype(str).str.strip()
    else:
        df["CustomerID"] = df["City"]

    # Zaman Boyutları
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Quarter"] = "Q" + df["InvoiceDate"].dt.quarter.astype(str)
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["YearWeek"] = df["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time.strftime("%Y-%m-%d"))
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    return df


# ==============================================================================
# 3. YARDIMCI ANALİTİK FONKSİYONLAR (DÖNEMSEL KIYAS, RFM, ABC, TAHMİNLEME)
# ==============================================================================
def format_currency_tr(val: float) -> str:
    """Büyük para birimlerini okunabilir ₺ formatına çevirir."""
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000:
        return f"{sign}₺{abs_val / 1_000_000:,.2f}M"
    elif abs_val >= 1_000:
        return f"{sign}₺{abs_val:,.0f}"
    return f"{sign}₺{abs_val:,.2f}"


def compute_period_deltas(df: pd.DataFrame) -> dict:
    """
    Seçilen tarih aralığını iki eşit yarıya (Önceki Dönem vs Son Dönem) bölerek
    temel KPI'ların büyüme / değişim yüzdelerini hesaplar.
    """
    if df.empty:
        return {"rev_delta": 0.0, "order_delta": 0.0, "cust_delta": 0.0, "aov_delta": 0.0, "qty_delta": 0.0}

    min_dt = df["InvoiceDate"].min()
    max_dt = df["InvoiceDate"].max()
    mid_dt = min_dt + (max_dt - min_dt) / 2

    prev_df = df[df["InvoiceDate"] < mid_dt]
    curr_df = df[df["InvoiceDate"] >= mid_dt]

    def safe_pct(curr, prev):
        if prev == 0 or pd.isna(prev):
            return 0.0
        return ((curr - prev) / abs(prev)) * 100.0

    prev_rev = prev_df["TotalAmount"].sum()
    curr_rev = curr_df["TotalAmount"].sum()

    prev_ord = prev_df["InvoiceNo"].nunique()
    curr_ord = curr_df["InvoiceNo"].nunique()

    prev_cust = prev_df["CustomerID"].nunique()
    curr_cust = curr_df["CustomerID"].nunique()

    prev_aov = prev_rev / prev_ord if prev_ord > 0 else 0
    curr_aov = curr_rev / curr_ord if curr_ord > 0 else 0

    prev_qty = prev_df["Quantity"].sum()
    curr_qty = curr_df["Quantity"].sum()

    return {
        "rev_delta": safe_pct(curr_rev, prev_rev),
        "order_delta": safe_pct(curr_ord, prev_ord),
        "cust_delta": safe_pct(curr_cust, prev_cust),
        "aov_delta": safe_pct(curr_aov, prev_aov),
        "qty_delta": safe_pct(curr_qty, prev_qty),
    }


@st.cache_data(show_spinner=False)
def compute_executive_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Hızlı B2B Bayi (Müşteri) RFM segmentasyonu hesaplar."""
    sales_df = df[df["TotalAmount"] > 0]
    if sales_df.empty:
        return pd.DataFrame()

    ref_date = sales_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        sales_df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
            City=("City", "first"),
            TopCategory=("Category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Genel"),
        )
        .reset_index()
    )

    if len(rfm) < 4:
        rfm["Segment"] = "Aktif Bayi"
        return rfm

    r_Score = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f_Score = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m_Score = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_Score"] = r_Score + f_Score + m_Score

    def assign_segment(row):
        r, f, m = r_Score.loc[row.name], f_Score.loc[row.name], m_Score.loc[row.name]
        if r >= 3 and f >= 3 and m >= 3:
            return "🏆 Şampiyon Bayiler (VIP)"
        elif f >= 3 and m >= 2:
            return "🤝 Sadık B2B Bayiler"
        elif r >= 3 and f <= 2:
            return "🌱 Yeni / Gelişen Bayiler"
        elif r == 2:
            return "⚠️ İlgi Bekleyen Bayiler"
        else:
            return "🚨 Kayıp Riski (Churn Risk)"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


@st.cache_data(show_spinner=False)
def compute_abc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Ürün (SKU) bazında Pareto / ABC envanter sınıflandırması yapar."""
    sales_df = df[df["TotalAmount"] > 0]
    if sales_df.empty:
        return pd.DataFrame()

    sku_df = (
        sales_df.groupby(["Description", "Category"], as_index=False)
        .agg(TotalRevenue=("TotalAmount", "sum"), TotalQty=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"))
        .sort_values("TotalRevenue", ascending=False)
    )
    total_rev = sku_df["TotalRevenue"].sum()
    sku_df["CumShare"] = (sku_df["TotalRevenue"].cumsum() / total_rev) * 100.0 if total_rev > 0 else 0

    sku_df["ABC_Class"] = np.where(
        sku_df["CumShare"] <= 80.0,
        "A Sınıfı (%80 Ciro - Kritik)",
        np.where(sku_df["CumShare"] <= 95.0, "B Sınıfı (%15 Ciro - Orta)", "C Sınıfı (%5 Ciro - Uzun Kuyruk)"),
    )
    return sku_df


def generate_revenue_forecast(ts_df: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """
    Zaman serisi üzerinde doğrusal trend + mevsimsel/hareketli ortalama bileşenli
    kısa vadeli tahmin ve %90 güven aralığı üretir.
    """
    if len(ts_df) < 5:
        return pd.DataFrame()

    ts_df = ts_df.sort_values("PeriodDate").reset_index(drop=True)
    y = ts_df["TotalAmount"].values.astype(float)
    x = np.arange(len(y))

    # Trend (1. derece sağlam doğrusal uyum)
    coeffs = np.polyfit(x, y, deg=1)
    trend = np.polyval(coeffs, x)
    residuals = y - trend
    std_err = np.std(residuals) if len(residuals) > 2 else np.mean(y) * 0.1

    # Haftalık / dönemsel döngüsel bileşen
    cycle_len = min(7, max(2, len(y) // 3))
    seasonal = np.array([np.mean(residuals[i::cycle_len]) for i in range(cycle_len)])

    last_date = pd.to_datetime(ts_df["PeriodDate"].iloc[-1])
    if len(ts_df) >= 2:
        freq_days = max(1, int((pd.to_datetime(ts_df["PeriodDate"].iloc[-1]) - pd.to_datetime(ts_df["PeriodDate"].iloc[-2])).days))
    else:
        freq_days = 1

    future_x = np.arange(len(y), len(y) + horizon)
    future_dates = [last_date + pd.Timedelta(days=freq_days * (i + 1)) for i in range(horizon)]
    future_trend = np.polyval(coeffs, future_x)
    future_seasonal = np.array([seasonal[idx % cycle_len] for idx in future_x])

    # Son gözlemlere ağırlık veren sönümlü tahmin
    recent_mean = np.mean(y[-min(7, len(y)) :])
    raw_forecast = 0.65 * (future_trend + future_seasonal) + 0.35 * recent_mean
    forecast_vals = np.maximum(0.0, raw_forecast)

    # Güven aralığı (ufuk uzadıkça genişleyen bant)
    ci_width = 1.645 * std_err * np.sqrt(1 + np.arange(1, horizon + 1) * 0.08)
    upper = forecast_vals + ci_width
    lower = np.maximum(0.0, forecast_vals - ci_width)

    return pd.DataFrame(
        {
            "PeriodDate": future_dates,
            "Forecast": forecast_vals,
            "UpperCI": upper,
            "LowerCI": lower,
        }
    )


def render_kpi_card(title: str, value: str, delta_val: float, sub_label: str, color_class: str = ""):
    """Özel tasarım HTML KPI kartı oluşturur."""
    if delta_val > 0.2:
        delta_html = f'<span class="delta-up">▲ %{delta_val:.1f}</span>'
    elif delta_val < -0.2:
        delta_html = f'<span class="delta-down">▼ %{abs(delta_val):.1f}</span>'
    else:
        delta_html = '<span class="delta-neutral">● Stabil</span>'

    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">
                {delta_html}
                <span>{sub_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 4. ANA YÖNETİCİ ÖZETİ EKRANI (RENDER FONKSİYONU)
# ==============================================================================
def render_executive_summary(filtered_df: pd.DataFrame, raw_df: pd.DataFrame):
    """
    Gelişmiş Yönetici Özeti (Executive Summary) ilk sayfasını oluşturur.
    Hem doğrudan çalıştırılabilir hem de streamlit_app.py içerisinden çağrılabilir.
    """
    inject_executive_css()

    if filtered_df.empty:
        st.warning("Seçilen filtre kriterlerine uygun işlem kaydı bulunamadı. Lütfen sol menüden filtreleri genişletin.")
        return

    # --------------------------------------------------------------------------
    # A. ÜST YÖNETİCİ (HERO) BANNER VE KURUMSAL SAĞLIK ENDEKSİ
    # --------------------------------------------------------------------------
    min_d = pd.to_datetime(filtered_df["Date"].min()).strftime("%d.%m.%Y")
    max_d = pd.to_datetime(filtered_df["Date"].max()).strftime("%d.%m.%Y")
    active_cities_count = filtered_df["City"].nunique()
    active_cat_count = filtered_df["Category"].nunique()
    total_records = len(filtered_df)

    st.markdown(
        f"""
        <div class="exec-hero">
            <div>
                <div class="exec-hero-title">🏛️ B2B Kitap Dağıtım — Üst Yönetici Özeti (Executive Summary)</div>
                <div class="exec-hero-subtitle">
                    Satış, Pazarlama ve Tedarik Zinciri Departmanları İçin Entegre C-Level Karar Destek ve Erken Uyarı Paneli
                </div>
            </div>
            <div class="exec-badge-group">
                <div class="exec-badge">📅 Dönem: {min_d} – {max_d}</div>
                <div class="exec-badge">🏙️ {active_cities_count} Aktif Şehir</div>
                <div class="exec-badge">📚 {active_cat_count} Kategori</div>
                <div class="exec-badge">🧾 {total_records:,} İşlem Satırı</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # B. STRATEJİK 6'LI C-LEVEL KPI KARTLARI VE DÖNEMSEL BÜYÜME (Δ)
    # --------------------------------------------------------------------------
    deltas = compute_period_deltas(filtered_df)

    net_revenue = filtered_df["TotalAmount"].sum()
    gross_revenue = filtered_df["GrossAmount"].sum()
    total_discount = filtered_df["DiscountAmount"].sum()
    discount_rate = (total_discount / gross_revenue * 100.0) if gross_revenue > 0 else 0.0

    total_orders = filtered_df["InvoiceNo"].nunique()
    aov = net_revenue / total_orders if total_orders > 0 else 0.0

    total_customers = filtered_df["CustomerID"].nunique()
    arpu = net_revenue / total_customers if total_customers > 0 else 0.0

    total_qty = filtered_df["Quantity"].sum()
    unique_skus = filtered_df["Description"].nunique()

    # İade / İptal oranı (Ham veri üzerinden aynı tarih ve şehir filtresiyle)
    raw_in_scope = raw_df[
        (raw_df["Date"] >= filtered_df["Date"].min())
        & (raw_df["Date"] <= filtered_df["Date"].max())
        & (raw_df["City"].isin(filtered_df["City"].unique()))
    ]
    return_invoices = raw_in_scope.loc[raw_in_scope["IsReturn"], "InvoiceNo"].nunique()
    all_invoices_scope = max(1, raw_in_scope["InvoiceNo"].nunique())
    return_rate = (return_invoices / all_invoices_scope) * 100.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_kpi_card(
            "TOPLAM NET CİRO",
            format_currency_tr(net_revenue),
            deltas["rev_delta"],
            "dönemsel büyüme",
            color_class="",
        )
    with k2:
        render_kpi_card(
            "ORT. SEPET (AOV)",
            format_currency_tr(aov),
            deltas["aov_delta"],
            f"{total_orders:,} fatura",
            color_class="emerald",
        )
    with k3:
        render_kpi_card(
            "AKTİF B2B BAYİ",
            f"{total_customers:,}",
            deltas["cust_delta"],
            f"Bayi başı {format_currency_tr(arpu)}",
            color_class="indigo",
        )
    with k4:
        render_kpi_card(
            "DAĞITILAN KİTAP",
            f"{total_qty:,.0f}",
            deltas["qty_delta"],
            f"{unique_skus:,} aktif SKU",
            color_class="cyan",
        )
    with k5:
        render_kpi_card(
            "BRÜT CİRO & İSKONTO",
            format_currency_tr(gross_revenue),
            -discount_rate if discount_rate > 0 else 0.0,
            f"Ort. İskonto: %{discount_rate:.1f}",
            color_class="amber",
        )
    with k6:
        render_kpi_card(
            "İADE / İPTAL ORANI",
            f"%{return_rate:.2f}",
            0.0,
            f"{return_invoices:,} iptal fatura",
            color_class="rose",
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # C. OTOMATİK YÖNETİCİ İÇGÖRÜLERİ VE ERKEN UYARI BANDI (AI EXECUTIVE PULSE)
    # --------------------------------------------------------------------------
    city_rev = filtered_df.groupby("City")["TotalAmount"].sum().sort_values(ascending=False)
    top_city = city_rev.index[0] if not city_rev.empty else "-"
    top_city_share = (city_rev.iloc[0] / net_revenue * 100.0) if (not city_rev.empty and net_revenue > 0) else 0.0

    cat_rev = filtered_df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False)
    top_cat = cat_rev.index[0] if not cat_rev.empty else "-"
    top_cat_share = (cat_rev.iloc[0] / net_revenue * 100.0) if (not cat_rev.empty and net_revenue > 0) else 0.0

    cust_rev = filtered_df.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False)
    top10_cust_share = (cust_rev.head(10).sum() / net_revenue * 100.0) if (not cust_rev.empty and net_revenue > 0) else 0.0

    rfm_df = compute_executive_rfm(filtered_df)
    churn_risk_count = int((rfm_df["Segment"] == "🚨 Kayıp Riski (Churn Risk)").sum()) if not rfm_df.empty else 0
    vip_count = int((rfm_df["Segment"] == "🏆 Şampiyon Bayiler (VIP)").sum()) if not rfm_df.empty else 0

    ins_col1, ins_col2, ins_col3 = st.columns(3)
    with ins_col1:
        st.markdown(
            f"""
            <div class="insight-box success">
                <div class="insight-title">📈 Ticari Liderlik & Kategori Dominansı</div>
                <p class="insight-text">
                    Toplam cironun <b>%{top_city_share:.1f}</b>'i <b>{top_city}</b> bölgesinden sağlanmaktadır.
                    En yüksek gelir üreten ürün grubu <b>{top_cat}</b> (<b>%{top_cat_share:.1f}</b> pay) olup
                    ana stok yatırımlarında önceliklendirilmelidir.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ins_col2:
        risk_cls = "warning" if top10_cust_share > 35 else "insight-box"
        st.markdown(
            f"""
            <div class="insight-box {risk_cls}">
                <div class="insight-title">🎯 Bayi Konsantrasyonu & Sadakat Durumu</div>
                <p class="insight-text">
                    İlk 10 B2B bayi toplam cironun <b>%{top10_cust_share:.1f}</b>'ini oluşturmaktadır.
                    Portföyde <b>{vip_count} Şampiyon (VIP)</b> bayiye karşılık, yeniden aktivasyon kampanyası
                    gerektiren <b>{churn_risk_count} kayıp riskli</b> bayi tespit edilmiştir.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ins_col3:
        abc_df = compute_abc_summary(filtered_df)
        a_class_count = int((abc_df["ABC_Class"].str.startswith("A")).sum()) if not abc_df.empty else 0
        a_class_pct = (a_class_count / max(1, len(abc_df))) * 100.0
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="insight-title">🚚 Tedarik Zinciri & Pareto (80/20) Verimliliği</div>
                <p class="insight-text">
                    Toplam <b>{unique_skus:,}</b> aktif kitaptan yalnızca <b>{a_class_count:,}</b> adedi
                    (portföyün <b>%{a_class_pct:.1f}</b>'i) cironun <b>%80'ini (A Sınıfı)</b> üretmektedir.
                    Bu kritik kitaplarda sıfır stoksuz kalma (stock-out) politikası uygulanmalıdır.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # D. FİNANSAL YÖRÜNGE, HAREKETLİ ORTALAMA VE TAHMİNLEME (FORECAST)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">📊 Finansal Yörünge, Büyüme Trendi ve Kısa Vadeli Ciro Tahmini</div>', unsafe_allow_html=True)

    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.4, 1.2, 1.4])
    with ctrl_c1:
        time_grain = st.radio(
            "Zaman Çözünürlüğü",
            options=["Günlük", "Haftalık", "Aylık"],
            index=1,
            horizontal=True,
            key="exec_time_grain",
        )
    with ctrl_c2:
        forecast_horizon = st.slider(
            "Tahmin Ufku (Dönem Sayısı)",
            min_value=4,
            max_value=30,
            value=8,
            help="Seçilen zaman çözünürlüğüne göre ileriye dönük tahmin periyodu.",
            key="exec_forecast_horizon",
        )
    with ctrl_c3:
        show_target_line = st.checkbox(
            "Ortalama Hedef Eşiğini ve Zirve Noktalarını Göster",
            value=True,
            key="exec_show_target",
        )

    if time_grain == "Günlük":
        ts_grouped = (
            filtered_df.groupby("Date", as_index=False)
            .agg(TotalAmount=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"), Quantity=("Quantity", "sum"))
            .rename(columns={"Date": "PeriodDate"})
        )
        ma_window = 7
        ma_label = "7 Günlük Hareketli Ortalama"
    elif time_grain == "Haftalık":
        ts_grouped = (
            filtered_df.groupby("YearWeek", as_index=False)
            .agg(TotalAmount=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"), Quantity=("Quantity", "sum"))
            .rename(columns={"YearWeek": "PeriodDate"})
        )
        ma_window = 4
        ma_label = "4 Haftalık Hareketli Ortalama"
    else:
        ts_grouped = (
            filtered_df.groupby("YearMonth", as_index=False)
            .agg(TotalAmount=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"), Quantity=("Quantity", "sum"))
            .rename(columns={"YearMonth": "PeriodDate"})
        )
        ts_grouped["PeriodDate"] = pd.to_datetime(ts_grouped["PeriodDate"] + "-01")
        ma_window = 3
        ma_label = "3 Aylık Hareketli Ortalama"

    ts_grouped["PeriodDate"] = pd.to_datetime(ts_grouped["PeriodDate"])
    ts_grouped = ts_grouped.sort_values("PeriodDate").reset_index(drop=True)
    ts_grouped["MovingAvg"] = ts_grouped["TotalAmount"].rolling(window=ma_window, min_periods=1).mean()

    forecast_df = generate_revenue_forecast(ts_grouped, horizon=forecast_horizon)

    fig_traj = make_subplots(specs=[[{"secondary_y": True}]])

    # Sipariş Hacmi (Arka plan çubuk)
    fig_traj.add_trace(
        go.Bar(
            x=ts_grouped["PeriodDate"],
            y=ts_grouped["Orders"],
            name="Sipariş Sayısı (Fatura)",
            marker_color="rgba(148, 163, 184, 0.28)",
            hovertemplate="Tarih: %{x|%Y-%m-%d}<br>Sipariş: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Gerçekleşen Net Ciro
    fig_traj.add_trace(
        go.Scatter(
            x=ts_grouped["PeriodDate"],
            y=ts_grouped["TotalAmount"],
            mode="lines+markers",
            name="Gerçekleşen Net Ciro (₺)",
            line=dict(color="#1D4ED8", width=2.5),
            marker=dict(size=5),
            hovertemplate="Tarih: %{x|%Y-%m-%d}<br>Net Ciro: ₺%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Hareketli Ortalama
    fig_traj.add_trace(
        go.Scatter(
            x=ts_grouped["PeriodDate"],
            y=ts_grouped["MovingAvg"],
            mode="lines",
            name=ma_label,
            line=dict(color="#F59E0B", width=2.5, dash="dot"),
            hovertemplate="Tarih: %{x|%Y-%m-%d}<br>Hareketli Ort.: ₺%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Tahmin (Forecast) ve Güven Aralığı
    if not forecast_df.empty:
        fig_traj.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["PeriodDate"], forecast_df["PeriodDate"][::-1]]),
                y=pd.concat([forecast_df["UpperCI"], forecast_df["LowerCI"][::-1]]),
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="%90 Tahmin Güven Bandı",
            ),
            secondary_y=False,
        )
        fig_traj.add_trace(
            go.Scatter(
                x=forecast_df["PeriodDate"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                name=f"Tahminlenen Ciro (+{forecast_horizon} Dönem)",
                line=dict(color="#10B981", width=2.8, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="Tahmin Tarihi: %{x|%Y-%m-%d}<br>Beklenen Ciro: ₺%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    # Ortalama hedef çizgisi ve zirve işaretlemesi
    if show_target_line and not ts_grouped.empty:
        mean_rev = ts_grouped["TotalAmount"].mean()
        target_rev = mean_rev * 1.10  # %10 büyüme hedefi
        fig_traj.add_hline(
            y=target_rev,
            line_dash="dashdot",
            line_color="#EF4444",
            annotation_text=f"Hedef Eşiği (+%10): {format_currency_tr(target_rev)}",
            annotation_position="top left",
            secondary_y=False,
        )

    fig_traj.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_traj.update_yaxes(title_text="Net Ciro (₺)", secondary_y=False, showgrid=True, gridcolor="#F1F5F9")
    fig_traj.update_yaxes(title_text="Sipariş Sayısı", secondary_y=True, showgrid=False)

    st.plotly_chart(fig_traj, use_container_width=True)

    # --------------------------------------------------------------------------
    # E. COĞRAFİ PARETO (80/20) VE KATEGORİ HİYERARŞİSİ (TREEMAP)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">🗺️ Bölgesel Ciro Konsantrasyonu (Pareto 80/20) ve Kategori Mimari Haritası</div>', unsafe_allow_html=True)

    geo_col, cat_col = st.columns([1.05, 1.0])

    with geo_col:
        city_pareto = (
            filtered_df.groupby("City", as_index=False)
            .agg(Revenue=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"))
            .sort_values("Revenue", ascending=False)
        )
        total_city_rev = city_pareto["Revenue"].sum()
        city_pareto["CumPct"] = (city_pareto["Revenue"].cumsum() / total_city_rev * 100.0) if total_city_rev > 0 else 0
        top_n_cities = city_pareto.head(15)

        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(
            go.Bar(
                x=top_n_cities["City"],
                y=top_n_cities["Revenue"],
                name="Şehir Cirosu (₺)",
                marker_color="#2563EB",
                hovertemplate="<b>%{x}</b><br>Ciro: ₺%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_pareto.add_trace(
            go.Scatter(
                x=top_n_cities["City"],
                y=top_n_cities["CumPct"],
                name="Kümülatif Pay (%)",
                mode="lines+markers",
                line=dict(color="#F59E0B", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x}</b><br>Kümülatif Ciro Payı: %%{y:.1f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_pareto.add_hline(
            y=80,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="%80 Pareto Sınırı",
            secondary_y=True,
        )
        fig_pareto.update_layout(
            title="🏙️ En Yüksek Ciro Üreten 15 Şehir & Pareto Eğrisi",
            template="plotly_white",
            height=410,
            margin=dict(l=15, r=15, t=45, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        fig_pareto.update_yaxes(title_text="Net Ciro (₺)", secondary_y=False)
        fig_pareto.update_yaxes(title_text="Kümülatif Pay (%)", range=[0, 105], secondary_y=True, showgrid=False)
        st.plotly_chart(fig_pareto, use_container_width=True)

    with cat_col:
        # Kategori -> En Çok Satan Kitaplar Hiyerarşik Treemap
        cat_book_df = (
            filtered_df[filtered_df["TotalAmount"] > 0]
            .groupby(["Category", "Description"], as_index=False)
            .agg(Revenue=("TotalAmount", "sum"), Quantity=("Quantity", "sum"))
        )
        # Görsel okunabilirlik için en yüksek cirolu 12 kategori ve her kategorinin ilk 6 kitabı
        top_cats = (
            cat_book_df.groupby("Category")["Revenue"]
            .sum()
            .nlargest(12)
            .index
        )
        treemap_df = (
            cat_book_df[cat_book_df["Category"].isin(top_cats)]
            .sort_values(["Category", "Revenue"], ascending=[True, False])
            .groupby("Category")
            .head(6)
        )

        if not treemap_df.empty:
            fig_tree = px.treemap(
                treemap_df,
                path=[px.Constant("Tüm Kategoriler"), "Category", "Description"],
                values="Revenue",
                color="Quantity",
                color_continuous_scale="Blues",
                title="📚 Kategori ve Lider Kitaplar Ciro Haritası (Treemap)",
                labels={"Revenue": "Net Ciro (₺)", "Quantity": "Satış Adedi"},
            )
            fig_tree.update_layout(
                height=410,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            fig_tree.update_traces(
                hovertemplate="<b>%{label}</b><br>Ciro: ₺%{value:,.2f}<br>Adet: %{color:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Treemap oluşturmak için yeterli pozitif satış verisi bulunamadı.")

    # --------------------------------------------------------------------------
    # F. DEPARTMANLAR ARASI STRATEJİK SEKMELER & WHAT-IF KARAR SİMÜLATÖRÜ
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">🧭 Departman Bazlı Yönetici Matrisleri & Karar Simülatörü</div>', unsafe_allow_html=True)

    exec_tab1, exec_tab2, exec_tab3, exec_tab4 = st.tabs(
        [
            "🎯 Pazarlama & B2B Bayi Portföyü (RFM)",
            "🚚 Tedarik Zinciri & ABC Envanter Özeti",
            "📅 Operasyonel Yoğunluk & Sipariş Ritmi",
            "🎛️ What-If (Senaryo & Karlılık) Simülatörü",
        ]
    )

    # --- TAB 1: PAZARLAMA & RFM ÖZETİ ---
    with exec_tab1:
        if not rfm_df.empty:
            rfm_c1, rfm_c2 = st.columns([1.0, 1.2])
            with rfm_c1:
                seg_summary = (
                    rfm_df.groupby("Segment", as_index=False)
                    .agg(
                        BayiSayisi=("CustomerID", "count"),
                        ToplamCiro=("Monetary", "sum"),
                        OrtRecency=("Recency", "mean"),
                        OrtFrekans=("Frequency", "mean"),
                    )
                    .sort_values("ToplamCiro", ascending=False)
                )
                fig_rfm = px.bar(
                    seg_summary,
                    x="ToplamCiro",
                    y="Segment",
                    orientation="h",
                    color="Segment",
                    text="BayiSayisi",
                    title="Segment Bazlı Ciro Katkısı ve Bayi Sayısı",
                    labels={"ToplamCiro": "Toplam Ciro (₺)", "Segment": "RFM Segmenti"},
                    color_discrete_sequence=["#1D4ED8", "#10B981", "#06B6D4", "#F59E0B", "#EF4444"],
                    template="plotly_white",
                )
                fig_rfm.update_traces(
                    texttemplate="%{text} Bayi",
                    textposition="inside",
                )
                fig_rfm.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_rfm, use_container_width=True)

            with rfm_c2:
                st.markdown("##### 🏆 En Değerli 10 B2B Bayi (Müşteri Liderlik Tablosu)")
                top_dealers = (
                    rfm_df.sort_values("Monetary", ascending=False)
                    .head(10)[["CustomerID", "City", "Segment", "Frequency", "Recency", "Monetary", "TopCategory"]]
                    .rename(
                        columns={
                            "CustomerID": "Bayi Kodu",
                            "City": "Şehir",
                            "Segment": "RFM Segmenti",
                            "Frequency": "Sipariş Sayısı",
                            "Recency": "Son Sipariş (Gün)",
                            "Monetary": "Toplam Ciro (₺)",
                            "TopCategory": "Favori Kategori",
                        }
                    )
                )
                st.dataframe(
                    top_dealers.style.format({"Toplam Ciro (₺)": "₺{:,.2f}", "Sipariş Sayısı": "{:,}", "Son Sipariş (Gün)": "{:,}"}),
                    use_container_width=True,
                    height=320,
                )
        else:
            st.info("RFM analizi için yeterli müşteri verisi bulunmuyor.")

    # --- TAB 2: TEDARİK ZİNCİRİ & ABC ENVANTER ÖZETİ ---
    with exec_tab2:
        if not abc_df.empty:
            abc_c1, abc_c2 = st.columns([1.0, 1.2])
            with abc_c1:
                abc_grp = (
                    abc_df.groupby("ABC_Class", as_index=False)
                    .agg(
                        SKU_Count=("Description", "count"),
                        Revenue=("TotalRevenue", "sum"),
                        Quantity=("TotalQty", "sum"),
                    )
                    .sort_values("ABC_Class")
                )
                fig_abc = px.pie(
                    abc_grp,
                    names="ABC_Class",
                    values="SKU_Count",
                    hole=0.48,
                    title="ABC Sınıflarına Göre Kitap (SKU) Adet Dağılımı",
                    color="ABC_Class",
                    color_discrete_map={
                        "A Sınıfı (%80 Ciro - Kritik)": "#10B981",
                        "B Sınıfı (%15 Ciro - Orta)": "#3B82F6",
                        "C Sınıfı (%5 Ciro - Uzun Kuyruk)": "#94A3B8",
                    },
                )
                fig_abc.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_abc, use_container_width=True)

            with abc_c2:
                st.markdown("##### 📦 A Sınıfı Kritik Kitaplar (En Yüksek Ciro Üreten İlk 10 Eser)")
                top_skus = (
                    abc_df.head(10)[["Description", "Category", "TotalQty", "Orders", "TotalRevenue", "CumShare"]]
                    .rename(
                        columns={
                            "Description": "Kitap Adı",
                            "Category": "Kategori",
                            "TotalQty": "Satış Adedi",
                            "Orders": "Fatura Sayısı",
                            "TotalRevenue": "Toplam Ciro (₺)",
                            "CumShare": "Kümülatif Ciro (%)",
                        }
                    )
                )
                st.dataframe(
                    top_skus.style.format(
                        {
                            "Satış Adedi": "{:,.0f}",
                            "Fatura Sayısı": "{:,}",
                            "Toplam Ciro (₺)": "₺{:,.2f}",
                            "Kümülatif Ciro (%)": "%{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    height=320,
                )
        else:
            st.info("ABC analizi için yeterli ürün verisi bulunmuyor.")

    # --- TAB 3: OPERASYONEL YOĞUNLUK MATRİSİ (HEATMAP) ---
    with exec_tab3:
        op_c1, op_c2 = st.columns([1.2, 1.0])
        with op_c1:
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_tr = {
                "Monday": "Pazartesi",
                "Tuesday": "Salı",
                "Wednesday": "Çarşamba",
                "Thursday": "Perşembe",
                "Friday": "Cuma",
                "Saturday": "Cumartesi",
                "Sunday": "Pazar",
            }
            heat_df = (
                filtered_df.groupby(["YearMonth", "DayOfWeek"], as_index=False)["TotalAmount"]
                .sum()
            )
            heat_piv = (
                heat_df.pivot(index="DayOfWeek", columns="YearMonth", values="TotalAmount")
                .reindex(dow_order)
                .fillna(0.0)
            )
            heat_piv.index = [dow_tr.get(d, d) for d in heat_piv.index]

            fig_heat = px.imshow(
                heat_piv,
                labels=dict(x="Ay (Yıl-Ay)", y="Haftanın Günü", color="Ciro (₺)"),
                color_continuous_scale="YlGnBu",
                aspect="auto",
                title="🔥 Aylara ve Haftanın Günlerine Göre Sevkiyat / Ciro Yoğunluk Matrisi",
            )
            fig_heat.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

        with op_c2:
            q_summary = (
                filtered_df.groupby("Quarter", as_index=False)
                .agg(Revenue=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"), Qty=("Quantity", "sum"))
                .sort_values("Quarter")
            )
            fig_q = px.bar(
                q_summary,
                x="Quarter",
                y="Revenue",
                text="Orders",
                color="Revenue",
                color_continuous_scale="Blues",
                title="📆 Çeyreklik (Quarterly) Ciro ve Sipariş Dağılımı",
                labels={"Quarter": "Çeyrek Dönem", "Revenue": "Net Ciro (₺)", "Orders": "Sipariş"},
                template="plotly_white",
            )
            fig_q.update_traces(texttemplate="%{text:,} Sipariş", textposition="outside")
            fig_q.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_q, use_container_width=True)

    # --- TAB 4: WHAT-IF SENARYO SİMÜLATÖRÜ ---
    with exec_tab4:
        st.markdown(
            "Bu simülatör, üst yönetimin **fiyatlandırma, iskonto politikası, bayi kazanımı ve sepet büyütme** "
            "kararlarının toplam net ciro üzerindeki beklenen etkisini anlık olarak modeller."
        )
        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        with sim_c1:
            price_chg = st.slider("Birim Fiyat Değişimi (%)", -20.0, 30.0, 5.0, step=1.0, key="sim_price")
        with sim_c2:
            vol_chg = st.slider("Sipariş Sıklığı / Adet Artışı (%)", -20.0, 30.0, 8.0, step=1.0, key="sim_vol")
        with sim_c3:
            new_dealer_chg = st.slider("Yeni Bayi Kazanımı (%)", 0.0, 25.0, 5.0, step=1.0, key="sim_dealer")
        with sim_c4:
            return_reduction = st.slider("İade Oranı İyileştirmesi (%)", 0.0, 50.0, 15.0, step=5.0, key="sim_ret")

        # Senaryo Hesaplaması
        price_factor = 1.0 + (price_chg / 100.0)
        vol_factor = 1.0 + (vol_chg / 100.0)
        dealer_factor = 1.0 + (new_dealer_chg / 100.0)

        # İade azaltımından geri kazanılan ciro
        return_loss_abs = abs(raw_in_scope.loc[raw_in_scope["IsReturn"], "TotalAmount"].sum())
        recovered_return_rev = return_loss_abs * (return_reduction / 100.0)

        projected_rev = (net_revenue * price_factor * vol_factor * dealer_factor) + recovered_return_rev
        rev_diff = projected_rev - net_revenue
        rev_diff_pct = (rev_diff / net_revenue * 100.0) if net_revenue > 0 else 0.0

        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Mevcut Net Ciro", format_currency_tr(net_revenue))
        res_c2.metric(
            "Simüle Edilen Hedef Ciro",
            format_currency_tr(projected_rev),
            delta=f"%{rev_diff_pct:+.2f} ({format_currency_tr(rev_diff)})",
        )
        res_c3.metric(
            "İade İyileştirmesinden Ek Kazanç",
            format_currency_tr(recovered_return_rev),
            delta=f"-%{return_reduction:.0f} İade Kaybı",
        )

        # Şelale (Waterfall) Grafiği ile Etki Analizi
        effect_price = net_revenue * (price_factor - 1.0)
        effect_vol = (net_revenue * price_factor) * (vol_factor - 1.0)
        effect_dealer = (net_revenue * price_factor * vol_factor) * (dealer_factor - 1.0)

        fig_wf = go.Figure(
            go.Waterfall(
                name="Senaryo Etkisi",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=[
                    "Mevcut Net Ciro",
                    "Fiyat Etkisi",
                    "Hacim / Sıklık Etkisi",
                    "Yeni Bayi Etkisi",
                    "İade Azaltım Kazancı",
                    "Simüle Edilen Ciro",
                ],
                textposition="outside",
                text=[
                    format_currency_tr(net_revenue),
                    format_currency_tr(effect_price),
                    format_currency_tr(effect_vol),
                    format_currency_tr(effect_dealer),
                    format_currency_tr(recovered_return_rev),
                    format_currency_tr(projected_rev),
                ],
                y=[net_revenue, effect_price, effect_vol, effect_dealer, recovered_return_rev, 0],
                connector={"line": {"color": "#64748B"}},
                increasing={"marker": {"color": "#10B981"}},
                decreasing={"marker": {"color": "#EF4444"}},
                totals={"marker": {"color": "#1E3A8A"}},
            )
        )
        fig_wf.update_layout(
            title="📊 What-If Senaryo Bileşenlerinin Ciroya Katkı Şelalesi (Waterfall)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # --------------------------------------------------------------------------
    # G. YÖNETİCİ ÖZET TABLOSU VE RAPOR İNDİRME (EXPORT)
    # --------------------------------------------------------------------------
    with st.expander("📥 Yönetici Özeti Şehir & Kategori Performans Tablosunu İncele / İndir"):
        exec_table = (
            filtered_df.groupby(["City", "Category"], as_index=False)
            .agg(
                SiparisSayisi=("InvoiceNo", "nunique"),
                AktifBayi=("CustomerID", "nunique"),
                ToplamAdet=("Quantity", "sum"),
                BrutCiro=("GrossAmount", "sum"),
                NetCiro=("TotalAmount", "sum"),
            )
            .sort_values("NetCiro", ascending=False)
        )
        st.dataframe(
            exec_table.head(100).style.format(
                {
                    "SiparisSayisi": "{:,}",
                    "AktifBayi": "{:,}",
                    "ToplamAdet": "{:,.0f}",
                    "BrutCiro": "₺{:,.2f}",
                    "NetCiro": "₺{:,.2f}",
                }
            ),
            use_container_width=True,
        )
        csv_bytes = exec_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📄 Yönetici Özet Raporunu İndir (CSV)",
            data=csv_bytes,
            file_name="B2B_Yonetici_Ozeti_Raporu.csv",
            mime="text/csv",
        )


# ==============================================================================
# 5. DOĞRUDAN ÇALIŞTIRMA BLOĞU (STANDALONE VE MULTI-PAGE DESTEĞİ)
# ==============================================================================
def main():
    with st.spinner("B2B Kitap Dağıtım Verisi Yükleniyor ve Yönetici Özeti Hazırlanıyor..."):
        raw_df = load_and_preprocess_data()

    # Kenar Çubuğu (Sidebar) Yönetici Filtreleri
    st.sidebar.image("https://img.icons8.com/color/96/books.png", width=60)
    st.sidebar.title("🏛️ Yönetici Filtreleri")
    st.sidebar.caption("Kapsamlı B2B Karar Destek Kontrolleri")

    include_returns = st.sidebar.toggle("İade / İptal İşlemlerini Dahil Et", value=False)
    if include_returns:
        active_df = raw_df.copy()
    else:
        active_df = raw_df[~raw_df["IsReturn"]].copy()

    # Tarih Aralığı
    min_date = active_df["Date"].min()
    max_date = active_df["Date"].max()
    date_range = st.sidebar.date_input(
        "📅 Analiz Dönemi",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        active_df = active_df[(active_df["Date"] >= start_d) & (active_df["Date"] <= end_d)]

    # Hızlı Bölge / Şehir Seçimi
    all_cities = sorted(active_df["City"].unique())
    top_cities_by_rev = (
        active_df.groupby("City")["TotalAmount"].sum().nlargest(15).index.tolist()
    )
    city_filter_mode = st.sidebar.radio(
        "🏙️ Şehir Kapsamı",
        options=["Tüm Türkiye", "En Yüksek Cirolu 15 Şehir", "Özel Şehir Seçimi"],
        index=0,
    )
    if city_filter_mode == "En Yüksek Cirolu 15 Şehir":
        active_df = active_df[active_df["City"].isin(top_cities_by_rev)]
    elif city_filter_mode == "Özel Şehir Seçimi":
        selected_cities = st.sidebar.multiselect(
            "Şehirleri Seçin",
            options=all_cities,
            default=top_cities_by_rev[:5],
        )
        if selected_cities:
            active_df = active_df[active_df["City"].isin(selected_cities)]

    # Kategori Filtresi
    all_categories = sorted(active_df["Category"].unique())
    selected_categories = st.sidebar.multiselect(
        "📚 Ürün Kategorisi (Opsiyonel)",
        options=all_categories,
        default=[],
        placeholder="Tüm Kategoriler",
    )
    if selected_categories:
        active_df = active_df[active_df["Category"].isin(selected_categories)]

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **İpucu:** Zaman çözünürlüğünü, tahmin ufkunu ve **What-If Karar Simülatörünü** "
        "sayfa içerisindeki interaktif kontrollerden yönetebilirsiniz."
    )

    # Yönetici Özeti Sayfasını Çiz
    render_executive_summary(active_df, raw_df)


if __name__ == "__main__":
    main()
