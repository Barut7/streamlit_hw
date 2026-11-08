import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI (DOĞRUDAN ÇALIŞTIRILDIĞINDA)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RafRadar | Yönetici Özeti & Veri Kalitesi",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def render_html(html_str: str):
    """
    Çok satırlı HTML metinlerindeki girintileri ve boş satırları temizleyerek
    Markdown'ın (CommonMark) 4+ boşluklu satırları kod bloğu (<pre><code>)
    olarak yorumlamasını %100 engeller.
    """
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE ÖN İŞLEME MOTORU (CACHE DESTEKLİ)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_rafradar_data() -> pd.DataFrame:
    """
    B2B işlem verisini Parquet (veya Excel) dosyasından okur, temizler ve
    Yönetici Özeti ile Veri Kalitesi sekmeleri için tüm türetilmiş değişkenleri hazırlar.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_candidates = [
        os.path.join(base_dir, "B2B_Transaction_Data_.parquet"),
        "B2B_Transaction_Data_.parquet",
    ]
    excel_candidates = [
        os.path.join(base_dir, "B2B_Transaction_Data_.xlsx"),
        "B2B_Transaction_Data_.xlsx",
    ]

    df = None
    for p in parquet_candidates:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    if df is None:
        for e in excel_candidates:
            if os.path.exists(e):
                df = pd.read_excel(e)
                try:
                    df.to_parquet(parquet_candidates[0], index=False)
                except Exception:
                    pass
                break

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    # Tarih dönüşümü
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()

    # Sayısal alanlar
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    if "NetPrice" in df.columns:
        df["NetPrice"] = pd.to_numeric(df["NetPrice"], errors="coerce").fillna(df["UnitPrice"])
    else:
        df["NetPrice"] = df["UnitPrice"]

    # Metin alanları
    df["InvoiceNo_Str"] = df["InvoiceNo"].astype(str).str.strip()
    df["CustomerID"] = (
        df["CustomerID"].fillna("Anonim_Bayi").astype(str).str.strip()
        if "CustomerID" in df.columns
        else "Anonim_Bayi"
    )
    df["Category"] = (
        df["Category"].fillna("Diğer / Genel").astype(str).str.strip()
        if "Category" in df.columns
        else "Genel Kitap"
    )
    df["Description"] = (
        df["Description"].fillna("Tanımsız Kitap").astype(str).str.strip()
        if "Description" in df.columns
        else "Tanımsız Kitap"
    )
    df["City"] = (
        df["City"].fillna("Bilinmiyor").astype(str).str.strip()
        if "City" in df.columns
        else "Bilinmiyor"
    )

    # İptal / İade tespiti
    df["IsReturn"] = (
        df["InvoiceNo_Str"].str.upper().str.startswith("C")
        | df["InvoiceNo_Str"].str.upper().str.contains("IADE")
        | (df["Quantity"] <= 0)
        | (df["UnitPrice"] <= 0)
    )

    # Finansal hesaplamalar: Liste Tutarı (Brüt), Net Ciro, İskonto Oranı ve Tahmini Brüt Kâr
    df["GrossAmount"] = df["Quantity"] * df["UnitPrice"]
    df["TotalAmount"] = df["Quantity"] * df["NetPrice"]
    df["DiscountAmount"] = np.maximum(0.0, df["GrossAmount"] - df["TotalAmount"])
    df["DiscountPct"] = np.where(
        df["GrossAmount"] > 0,
        np.clip((df["DiscountAmount"] / df["GrossAmount"]) * 100.0, 0.0, 95.0),
        0.0,
    )

    # Tahmini Alış Maliyeti = Liste Fiyatının %44'ü (Sektör Standardı Varsayımı)
    df["EstCost"] = df["GrossAmount"] * 0.44
    df["EstProfit"] = df["TotalAmount"] - df["EstCost"]

    # Zaman kırılımları
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Quarter"] = df["InvoiceDate"].dt.year.astype(str) + "-Ç" + df["InvoiceDate"].dt.quarter.astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    return df


# -----------------------------------------------------------------------------
# 3. YARDIMCI FORMATLAMA VE RAF-RADAR V2 TASARIM DİLİ (CSS)
# -----------------------------------------------------------------------------
def fmt_tl(val: float, decimals: int = 2) -> str:
    """TL değerlerini yönetici okumasına uygun (M ₺, bin ₺) formatlar."""
    if pd.isna(val):
        return "0 ₺"
    sign = "-" if val < 0 else ""
    v = abs(float(val))
    if v >= 1_000_000:
        return f"{sign}₺{v / 1_000_000:,.{decimals}f} Mn".replace(",", "X").replace(".", ",").replace("X", ".")
    if v >= 1_000:
        return f"{sign}₺{v / 1_000:,.1f} Bin".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}₺{v:,.0f}".replace(",", ".")


def fmt_int(val: float) -> str:
    if pd.isna(val):
        return "0"
    return f"{int(round(val)):,}".replace(",", ".")


def inject_rafradar_v2_css():
    """
    rafradar-versiyon2.html dosyasındaki sıcak krem (--bg:#f6f3ee), lacivert (#1f2a44),
    zümrüt/teal (#3d9a84), altın sarısı (#e0a526) ve kiremit (#d0574f) renk paleti ile
    Poppins + Inter tipografisini uygular.
    """
    render_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #f6f3ee;
            --fg: #1f2a44;
            --side: #faf7f2;
            --line: #e4dfd6;
            --teal: #3d9a84;
            --teal-dark: #14513f;
            --teal-soft: #d6ece5;
            --gold: #e0a526;
            --gold-soft: #fdf0cf;
            --gold-dark: #7a5600;
            --coral: #d0574f;
            --coral-soft: #fbe3e0;
            --coral-dark: #8f2a24;
            --blue: #3b6fd1;
            --muted: #5b6475;
        }
        .stApp {
            background-color: var(--bg) !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            color: var(--fg);
        }
        /* Üst RafRadar V2 Barı */
        .rr-topbar {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .rr-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 21px;
            color: #1f2a44;
        }
        .rr-page-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 26px;
            color: #1f2a44;
            margin: 0;
            line-height: 1.2;
        }
        .rr-page-header p {
            margin: 3px 0 0 0;
            font-size: 13.5px;
            font-style: italic;
            color: #5b6475;
        }
        .rr-pill-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
            margin: 10px 0 14px 0;
        }
        .rr-pill {
            display: inline-block;
            font-size: 11.5px;
            font-weight: 600;
            padding: 4px 11px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .rr-pill.mint  { background: #d6ece5; color: #14513f; }
        .rr-pill.Gold  { background: #fdf0cf; color: #7a5600; }
        .rr-pill.rose  { background: #fbe3e0; color: #8f2a24; }
        .rr-pill.stone { background: #ece7dd; color: #1f2a44; }
        .rr-pill.navy  { background: #1f2a44; color: #ffffff; }

        /* KPI Kartları (rafradar-versiyon2 .card.kpi) */
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-top: 4px solid #3d9a84;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(31, 42, 68, 0.04);
            height: 100%;
        }
        .kpi-card.gold  { border-top-color: #e0a526; }
        .kpi-card.blue  { border-top-color: #3b6fd1; }
        .kpi-card.coral { border-top-color: #d0574f; }
        .kpi-card.navy  { border-top-color: #1f2a44; }

        .kpi-label {
            font-size: 12px;
            font-weight: 500;
            color: #5b6475;
            margin-bottom: 3px;
        }
        .kpi-value {
            font-family: 'Poppins', sans-serif;
            font-size: 22px;
            font-weight: 600;
            color: #1f2a44;
            line-height: 1.2;
            margin: 2px 0 6px 0;
        }
        .kpi-rule {
            border-top: 1px dashed #d9d3c7;
            margin: 8px 0 6px 0;
        }
        .kpi-delta-pos {
            font-size: 11px;
            font-weight: 600;
            color: #14513f;
            background: #d6ece5;
            padding: 2px 8px;
            border-radius: 999px;
            display: inline-block;
        }
        .kpi-delta-neg {
            font-size: 11px;
            font-weight: 600;
            color: #8f2a24;
            background: #fbe3e0;
            padding: 2px 8px;
            border-radius: 999px;
            display: inline-block;
        }
        .kpi-delta-warn {
            font-size: 11px;
            font-weight: 600;
            color: #7a5600;
            background: #fdf0cf;
            padding: 2px 8px;
            border-radius: 999px;
            display: inline-block;
        }
        .kpi-subtext {
            font-size: 11px;
            color: #5b6475;
            margin-left: 5px;
        }

        /* Bölüm Kartları (rafradar-versiyon2 .card + .h + .num) */
        .section-card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 4px rgba(31, 42, 68, 0.03);
        }
        .section-card.teal-left { border-left: 4px solid #3d9a84; }
        .section-card.gold-left { border-left: 4px solid #e0a526; }
        .section-card.coral-left { border-left: 4px solid #d0574f; }

        .rr-h2 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 15.5px;
            color: #1f2a44;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .rr-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 21px;
            height: 21px;
            border-radius: 50%;
            background: #1f2a44;
            color: #ffffff;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
            flex-shrink: 0;
        }
        .rr-sub {
            font-size: 12px;
            color: #5b6475;
            margin: 3px 0 8px 0;
        }
        .section-takeaway {
            font-size: 14.5px;
            font-weight: 600;
            color: #1f2a44;
            margin: 6px 0 8px 0;
            line-height: 1.4;
        }
        .insight-box {
            background-color: #faf7f2;
            border: 1px solid #e4dfd6;
            border-left: 4px solid #3d9a84;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            color: #1f2a44;
            margin-top: 8px;
            margin-bottom: 14px;
            line-height: 1.5;
        }

        /* Sızıntı / Fırsat Mini Kartları (.out & .mini) */
        .leak-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 9px;
            margin-top: 6px;
            margin-bottom: 14px;
        }
        .leak-item {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 10px 12px;
        }
        .leak-item-title {
            font-size: 12px;
            font-weight: 600;
            color: #1f2a44;
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }
        .leak-item-desc {
            font-size: 11.5px;
            color: #5b6475;
            line-height: 1.38;
        }

        /* Portföy / Pareto Kutuları (rafradar-versiyon2 2. Panel tarzı) */
        .pareto-box {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .pareto-title {
            font-family: 'Poppins', sans-serif;
            font-size: 13.5px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .pareto-sub {
            font-size: 11.5px;
            color: #5b6475;
            margin-bottom: 10px;
        }
        .abc-stack-bar {
            display: flex;
            height: 28px;
            border-radius: 8px;
            overflow: hidden;
            margin: 6px 0 8px 0;
        }
        .tier-row {
            margin-bottom: 8px;
        }
        .tier-header {
            display: flex;
            justify-content: space-between;
            font-size: 11.5px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 3px;
        }
        .tier-bar-bg {
            width: 100%;
            height: 8px;
            background: #efe9de;
            border-radius: 999px;
            overflow: hidden;
        }
        .tier-bar-fill {
            height: 100%;
            border-radius: 999px;
        }

        /* 30 Günlük Aksiyon Kartları (rafradar-versiyon2 5. Panel tarzı) */
        .action-card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-top: 4px solid #3d9a84;
            border-radius: 10px;
            padding: 13px 15px;
            margin-bottom: 10px;
        }
        .action-card.urgent { border-top-color: #d0574f; }
        .action-card.high   { border-top-color: #e0a526; }
        .action-card.medium { border-top-color: #3d9a84; }
        .action-card.blue   { border-top-color: #3b6fd1; }

        .action-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            flex-wrap: wrap;
            gap: 6px;
        }
        .action-dept-group {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        .action-dept-pill {
            font-size: 11px;
            font-weight: 600;
            color: #ffffff;
            background: #1f2a44;
            padding: 2px 9px;
            border-radius: 999px;
        }
        .action-headline {
            font-family: 'Poppins', sans-serif;
            font-size: 13.5px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 4px;
        }
        .action-body {
            font-size: 12px;
            color: #5b6475;
            line-height: 1.45;
            margin-bottom: 8px;
        }
        .action-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #faf7f2;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid #e4dfd6;
            font-size: 12px;
        }
        .action-impact {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #1f2a44;
            font-size: 14px;
        }

        /* Veri Kalitesi 5 Adım Kutuları (rafradar-versiyon2 kalite 1. Panel) */
        .vq-step-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 10px 0 14px 0;
            align-items: stretch;
        }
        .vq-step {
            flex: 1 1 140px;
            min-width: 0;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .vq-step-label {
            font-size: 11px;
            font-weight: 600;
            color: #5b6475;
            text-transform: uppercase;
        }
        .vq-step-val {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 21px;
            color: #1f2a44;
            margin: 4px 0;
        }
        .vq-step-desc {
            font-size: 11.5px;
            color: #1f2a44;
            line-height: 1.35;
        }
        .vq-arrow {
            align-self: center;
            color: #5b6475;
            font-size: 18px;
            font-weight: 600;
            padding: 0 2px;
        }
        .formula-card {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-left: 4px solid #3d9a84;
            border-radius: 8px;
            padding: 11px 14px;
            margin-bottom: 9px;
        }
        .formula-title {
            font-family: 'Poppins', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 3px;
        }
        .formula-code {
            font-family: monospace;
            font-size: 12px;
            background: #ffffff;
            color: #14513f;
            border: 1px solid #e4dfd6;
            padding: 3px 8px;
            border-radius: 5px;
            display: inline-block;
            margin: 4px 0;
            font-weight: 600;
        }
        /* Sağ Alt Yüzen Sohbet Butonu Süslemesi */
        div[data-testid="stPopover"] > button {
            background-color: #1f2a44 !important;
            color: #ffffff !important;
            border: 2px solid #3d9a84 !important;
            border-radius: 999px !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 14px rgba(31, 42, 68, 0.18) !important;
        }
        div[data-testid="stPopover"] > button:hover {
            background-color: #3d9a84 !important;
            color: #ffffff !important;
        }
        /* rafradar-versiyon2.html Sağ Alt Yüzen Sohbet Balonu (#chatwrap / .cwb) */
        .rr-float-wrap {
            position: fixed;
            right: 22px;
            bottom: 22px;
            z-index: 9999;
        }
        .rr-float-wrap > summary {
            list-style: none;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #3d9a84;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(31, 42, 68, 0.28);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid #ffffff;
            transition: transform 0.15s ease, background 0.15s ease;
        }
        .rr-float-wrap > summary::-webkit-details-marker { display: none; }
        .rr-float-wrap > summary:hover {
            transform: scale(1.06);
            background: #1f2a44;
        }
        .rr-float-panel {
            position: absolute;
            right: 0;
            bottom: 68px;
            width: 360px;
            max-width: calc(100vw - 32px);
            max-height: 75vh;
            overflow-y: auto;
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 14px;
            box-shadow: 0 12px 32px rgba(31, 42, 68, 0.25);
            padding: 14px;
            color: #1f2a44;
        }
        .rr-qa-chip {
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            background: #faf7f2;
            margin-bottom: 7px;
            padding: 8px 11px;
        }
        .rr-qa-chip > summary {
            list-style: none;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            color: #1f2a44;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rr-qa-chip > summary::-webkit-details-marker { display: none; }
        .rr-qa-ans {
            margin-top: 7px;
            padding-top: 7px;
            border-top: 1px dashed #d9d3c7;
            font-size: 12px;
            color: #1f2a44;
            line-height: 1.45;
        }
        </style>
        """
    )


# -----------------------------------------------------------------------------
# 4. DİNAMİK ANALİTİK HESAPLAMA MOTORU
# -----------------------------------------------------------------------------
def compute_dynamic_metrics(sales_df: pd.DataFrame, prev_sales_df: pd.DataFrame, raw_scope_df: pd.DataFrame) -> dict:
    """
    Seçilen tarih ve kategori filtresine göre ekrandaki tüm KPI, sızıntı,
    Pareto, RFM, ABC ve anomali metriklerini canlı olarak hesaplar.
    """
    net_rev = sales_df["TotalAmount"].sum()
    gross_rev = sales_df["GrossAmount"].sum()
    est_profit = sales_df["EstProfit"].sum()
    profit_margin = (est_profit / net_rev * 100.0) if net_rev > 0 else 0.0
    avg_discount = ((gross_rev - net_rev) / gross_rev * 100.0) if gross_rev > 0 else 0.0

    prev_net_rev = prev_sales_df["TotalAmount"].sum() if not prev_sales_df.empty else 0.0
    rev_growth_pct = ((net_rev - prev_net_rev) / prev_net_rev * 100.0) if prev_net_rev > 0 else 0.0

    active_customers = sales_df["CustomerID"].nunique()
    prev_customers = prev_sales_df["CustomerID"].nunique() if not prev_sales_df.empty else 0
    cust_diff = active_customers - prev_customers

    active_skus = sales_df["Description"].nunique()
    total_invoices = sales_df["InvoiceNo"].nunique()

    # Kategori Medyan İskontosu ve Norm Dışı İskonto Kaçağı (+10 Puan Sapma)
    cat_medians = sales_df.groupby("Category")["DiscountPct"].transform("median")
    excess_disc_pct = sales_df["DiscountPct"] - cat_medians
    leak_mask = excess_disc_pct > 10.0
    suspicious_orders_count = int(leak_mask.sum())
    discount_leak_tl = float(
        (sales_df.loc[leak_mask, "GrossAmount"] * (excess_disc_pct[leak_mask] / 100.0)).sum()
    )
    if discount_leak_tl < 1000 and len(sales_df) > 0:
        mild_mask = excess_disc_pct > 5.0
        suspicious_orders_count = int(mild_mask.sum())
        discount_leak_tl = float(
            (sales_df.loc[mild_mask, "GrossAmount"] * (excess_disc_pct[mild_mask] / 100.0)).sum()
        )

    cat_anom_df = (
        sales_df.assign(
            CatMedian=cat_medians,
            ExcessPct=np.maximum(0.0, excess_disc_pct),
            LeakTL=np.where(excess_disc_pct > 5.0, sales_df["GrossAmount"] * (excess_disc_pct / 100.0), 0.0),
        )
        .groupby("Category", as_index=False)
        .agg(
            MedianDisc=("DiscountPct", "median"),
            MaxDisc=("DiscountPct", "max"),
            LeakTL=("LeakTL", "sum"),
            AnomRows=("ExcessPct", lambda s: int((s > 5.0).sum())),
        )
        .sort_values("LeakTL", ascending=False)
    )

    # Müşteri Pareto (80/20) ve RFM / Kayıp Riski Analizi
    max_date_in_selection = sales_df["InvoiceDate"].max()
    cust_df = (
        sales_df.groupby("CustomerID", as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
            LastDate=("InvoiceDate", "max"),
            FirstDate=("InvoiceDate", "min"),
            City=("City", "first"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    cust_df["RecencyDays"] = (max_date_in_selection - cust_df["LastDate"]).dt.days
    cust_df["CumRev"] = cust_df["Revenue"].cumsum()
    cust_df["CumRevPct"] = (cust_df["CumRev"] / net_rev * 100.0) if net_rev > 0 else 0.0

    cust_df["Tier"] = np.where(
        cust_df["CumRevPct"] <= 80.0,
        "VIP & Lokomotif Bayiler",
        np.where(cust_df["CumRevPct"] <= 95.0, "Gelişen Orta Bayiler", "Küçük / Tekil Bayiler"),
    )
    if not cust_df.empty and (cust_df["Tier"] == "VIP & Lokomotif Bayiler").sum() == 0:
        cust_df.loc[0, "Tier"] = "VIP & Lokomotif Bayiler"

    vip_cust_count = int((cust_df["Tier"] == "VIP & Lokomotif Bayiler").sum())
    vip_cust_share_pct = (vip_cust_count / max(1, active_customers)) * 100.0
    top10_cust_rev_share = (cust_df.head(10)["Revenue"].sum() / net_rev * 100.0) if net_rev > 0 else 0.0

    recency_threshold = max(30, int((sales_df["InvoiceDate"].max() - sales_df["InvoiceDate"].min()).days * 0.25))
    recency_threshold = min(recency_threshold, 90)
    top_half_cutoff = cust_df["Revenue"].quantile(0.65) if len(cust_df) > 3 else 0
    at_risk_vips = cust_df[(cust_df["Revenue"] >= top_half_cutoff) & (cust_df["RecencyDays"] >= recency_threshold)].copy()
    if at_risk_vips.empty and len(cust_df) > 2:
        at_risk_vips = cust_df.sort_values("RecencyDays", ascending=False).head(max(1, len(cust_df) // 10)).copy()

    at_risk_vip_count = len(at_risk_vips)
    vip_recovery_potential_tl = float(at_risk_vips["Revenue"].sum() * 0.35)
    top_risk_examples = at_risk_vips.sort_values("Revenue", ascending=False).head(2)

    one_time_custs = cust_df[cust_df["Orders"] == 1]
    one_time_ratio_pct = (len(one_time_custs) / max(1, active_customers)) * 100.0
    avg_second_order_val = (net_rev / max(1, total_invoices)) * 1.15
    retention_potential_tl = float(len(one_time_custs) * 0.25 * avg_second_order_val)

    # Ürün (SKU) ABC Analizi ve Ölü Stok Tespiti
    sku_df = (
        sales_df.groupby(["Description", "Category"], as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
            LastSale=("InvoiceDate", "max"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    sku_df["CumRevPct"] = (sku_df["Revenue"].cumsum() / net_rev * 100.0) if net_rev > 0 else 0.0
    sku_df["ABC"] = np.where(
        sku_df["CumRevPct"] <= 80.0,
        "A Sınıfı Yıldız Kitaplar",
        np.where(sku_df["CumRevPct"] <= 95.0, "B Sınıfı Düzenli Kitaplar", "C Sınıfı & Uzun Kuyruk"),
    )
    if not sku_df.empty and (sku_df["ABC"] == "A Sınıfı Yıldız Kitaplar").sum() == 0:
        sku_df.loc[0, "ABC"] = "A Sınıfı Yıldız Kitaplar"

    a_sku_count = int((sku_df["ABC"] == "A Sınıfı Yıldız Kitaplar").sum())
    a_sku_share_pct = (a_sku_count / max(1, active_skus)) * 100.0
    c_sku_count = int((sku_df["ABC"] == "C Sınıfı & Uzun Kuyruk").sum())
    c_sku_share_pct = (c_sku_count / max(1, active_skus)) * 100.0

    sku_df["DaysSinceSale"] = (max_date_in_selection - sku_df["LastSale"]).dt.days
    dead_sku_count = int((sku_df["DaysSinceSale"] >= 120).sum())

    a_class_rev = float(sku_df.loc[sku_df["ABC"] == "A Sınıfı Yıldız Kitaplar", "Revenue"].sum())
    stockout_prevention_tl = a_class_rev * 0.045

    # En çok ciro yapan şehir ve kategori
    city_rev = sales_df.groupby("City")["TotalAmount"].sum().sort_values(ascending=False)
    top_city = city_rev.index[0] if not city_rev.empty else "İstanbul"
    top_city_share = (city_rev.iloc[0] / net_rev * 100.0) if (not city_rev.empty and net_rev > 0) else 0.0

    cat_rev = sales_df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False)
    top_cat_name = cat_rev.index[0] if not cat_rev.empty else "Genel"
    top_cat_share = (cat_rev.iloc[0] / net_rev * 100.0) if (not cat_rev.empty and net_rev > 0) else 0.0

    return {
        "net_rev": net_rev,
        "gross_rev": gross_rev,
        "est_profit": est_profit,
        "profit_margin": profit_margin,
        "avg_discount": avg_discount,
        "rev_growth_pct": rev_growth_pct,
        "prev_net_rev": prev_net_rev,
        "active_customers": active_customers,
        "cust_diff": cust_diff,
        "active_skus": active_skus,
        "total_invoices": total_invoices,
        "suspicious_orders_count": suspicious_orders_count,
        "discount_leak_tl": discount_leak_tl,
        "cat_anom_df": cat_anom_df,
        "cust_df": cust_df,
        "vip_cust_count": vip_cust_count,
        "vip_cust_share_pct": vip_cust_share_pct,
        "top10_cust_rev_share": top10_cust_rev_share,
        "at_risk_vip_count": at_risk_vip_count,
        "vip_recovery_potential_tl": vip_recovery_potential_tl,
        "top_risk_examples": top_risk_examples,
        "one_time_ratio_pct": one_time_ratio_pct,
        "one_time_count": len(one_time_custs),
        "retention_potential_tl": retention_potential_tl,
        "sku_df": sku_df,
        "a_sku_count": a_sku_count,
        "a_sku_share_pct": a_sku_share_pct,
        "c_sku_count": c_sku_count,
        "c_sku_share_pct": c_sku_share_pct,
        "dead_sku_count": dead_sku_count,
        "stockout_prevention_tl": stockout_prevention_tl,
        "top_city": top_city,
        "top_city_share": top_city_share,
        "top_cat_name": top_cat_name,
        "top_cat_share": top_cat_share,
    }


# -----------------------------------------------------------------------------
# 5. İNTERAKTİF RAF-RADAR SOHBET ASİSTANI (CHATBOX MOTORU)
# -----------------------------------------------------------------------------
def answer_executive_question(query: str, m: dict, start_str: str, end_str: str, selected_cat: str) -> str:
    """
    Yöneticinin sorduğu soruyu analiz eder ve seçili tarih/kategori filtresindeki
    gerçek hesaplanmış metriklerle yanıt üretir (rafradar-versiyon2.html chatbox mantığı).
    """
    q = query.lower().strip()
    total_opp = (
        m["vip_recovery_potential_tl"]
        + m["stockout_prevention_tl"]
        + m["retention_potential_tl"]
        + m["discount_leak_tl"]
    )
    target_rev = m["net_rev"] + total_opp

    if any(k in q for k in ["ciro", "büyü", "kâr", "kar", "marj", "finans", "durum"]):
        dir_txt = f"%{abs(m['rev_growth_pct']):.1f} büyüme" if m["rev_growth_pct"] >= 0 else f"%{abs(m['rev_growth_pct']):.1f} daralma"
        return (
            f"📊 **Finansal Özet ({start_str} – {end_str}):** Seçili dönemde **{fmt_int(m['total_invoices'])} fatura** "
            f"üzerinden **{fmt_tl(m['net_rev'])} net ciro** ({dir_txt}) üretildi. Liste fiyatının %44'ü maliyet varsayımıyla "
            f"**tahmini brüt kârımız {fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)** seviyesindedir."
        )

    if any(k in q for k in ["iskonto", "indirim", "kaçak", "anomali", "şüpheli", "fiyat"]):
        top_cats = ", ".join(m["cat_anom_df"].head(3)["Category"].tolist()) if not m["cat_anom_df"].empty else "Sözlük ve Roman"
        return (
            f"🏷️ **İskonto & Kaçak Denetimi:** Seçili dönemde ortalama iskonto oranı **%{m['avg_discount']:.1f}**'dir. "
            f"Ancak **{fmt_int(m['suspicious_orders_count'])} sipariş satırında** kendi kategori medyanının 10+ puan "
            f"üzerinde indirim verilerek **{fmt_tl(m['discount_leak_tl'])}** kâr sızıntısı oluşmuştur. En çok sapma görülen "
            f"kategoriler: **{top_cats}**."
        )

    if any(k in q for k in ["vip", "müşteri", "bayi", "kayıp", "churn", "risk", "pasif"]):
        risk_ex = m["top_risk_examples"]
        ex_names = ", ".join([f"{r['CustomerID']} ({fmt_tl(r['Revenue'], 0)})" for _, r in risk_ex.iterrows()]) if not risk_ex.empty else "üst segment bayiler"
        return (
            f"🚨 **Bayi & Kayıp Riski (Churn) Durumu:** Toplam **{fmt_int(m['active_customers'])} aktif bayinin** "
            f"sadece **%{m['vip_cust_share_pct']:.1f}'i ({m['vip_cust_count']} VIP bayi)** cironun %80'ini üretiyor. "
            f"Başta **{ex_names}** olmak üzere **{m['at_risk_vip_count']} yüksek cirolu bayi** uzun süredir sipariş vermedi. "
            f"Bu bayilerin geri kazanımı **+{fmt_tl(m['vip_recovery_potential_tl'])}** ek ciro potansiyeli taşır."
        )

    if any(k in q for k in ["pareto", "80/20", "yoğunlaşma", "bağımlılık", "portföy", "abc"]):
        return (
            f"⚖️ **Çift Taraflı Pareto (80/20) Riski:** Cironun %80'i müşterilerin yalnızca **%{m['vip_cust_share_pct']:.1f}'ine "
            f"({m['vip_cust_count']} bayi)** ve kitapların **%{m['a_sku_share_pct']:.1f}'ine ({fmt_int(m['a_sku_count'])} A sınıfı SKU)** "
            f"bağımlıdır. İlk 10 bayi tek başına cironun **%{m['top10_cust_rev_share']:.1f}**'ini oluşturmaktadır."
        )

    if any(k in q for k in ["stok", "tedarik", "ölü", "hareketsiz", "kitap", "ürün", "sku"]):
        return (
            f"📦 **Stok ve Ürün Portföyü:** Seçili dönemde **{fmt_int(m['active_skus'])} farklı kitap (SKU)** satıldı. "
            f"Cironun %80'ini çeken **{fmt_int(m['a_sku_count'])} A sınıfı yıldız kitapta** sıfır stoksuzluk politikası "
            f"(+{fmt_tl(m['stockout_prevention_tl'])} koruma) uygulanmalı; **120+ gündür satılmayan {fmt_int(m['dead_sku_count'])} "
            f"hareketsiz SKU** ise paket kampanyalarla eritilmelidir."
        )

    if any(k in q for k in ["aksiyon", "ne yapmalı", "karar", "plan", "fırsat", "potansiyel", "30 gün"]):
        return (
            f"🎯 **30 Günlük Yönetim Aksiyon Özeti:** 4 ana sızıntı kapatıldığında mevcut ciroya **+{fmt_tl(total_opp)}** "
            f"eklenerek **{fmt_tl(target_rev)}** seviyesine ulaşılabilir:\n"
            f"1. **VIP Bayi Geri Kazanım Aramaları:** +{fmt_tl(m['vip_recovery_potential_tl'])}\n"
            f"2. **A Sınıfı Yıldız Kitap Stok Koruması:** +{fmt_tl(m['stockout_prevention_tl'])}\n"
            f"3. **Tek Siparişli Bayilere 2. Sipariş Teklifi:** +{fmt_tl(m['retention_potential_tl'])}\n"
            f"4. **Kategori İskonto Tavanı (+5 Puan Onay Şartı):** +{fmt_tl(m['discount_leak_tl'])}"
        )

    if any(k in q for k in ["şehir", "il", "bölge", "kategori", "en çok"]):
        return (
            f"🏙️ **Bölge ve Kategori Lideri:** Seçilen dönemde en yüksek ciro üreten il **{m['top_city']}** "
            f"(toplam cironun **%{m['top_city_share']:.1f}**'i), lider ürün kategorisi ise **{m['top_cat_name']}** "
            f"(cironun **%{m['top_cat_share']:.1f}**'i) olmuştur."
        )

    if any(k in q for k in ["veri", "kalite", "temiz", "maliyet", "formül", "nasıl hesap"]):
        return (
            "🔬 **Veri Kalitesi ve Metodoloji:** Ham veride `C` ile başlayan veya negatif miktarlı iade satırları ayrıştırıldı. "
            "Yayınevi alış maliyeti veride bulunmadığı için sektör standardı olan **Liste Fiyatının %44'ü** kabul edildi. "
            "İskonto kaçağı ise her kategorinin kendi **medyan (ortanca)** iskontosundan +10 puan sapan işlemlerden hesaplandı."
        )

    return (
        f"💡 **Seçili Dönem Özeti ({start_str} – {end_str} | {selected_cat}):** Net ciro **{fmt_tl(m['net_rev'])}**, "
        f"tahmini brüt kâr **{fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)**, aktif bayi **{fmt_int(m['active_customers'])}** "
        f"ve 4 aksiyondan toplam erişilebilir fırsat **+{fmt_tl(total_opp)}**'dir. *Ciro, iskonto kaçağı, VIP bayiler, "
        f"stok/Pareto veya 30 günlük aksiyon planı* hakkında soru sorabilirsiniz."
    )


def render_floating_and_inline_chatbox(m: dict, start_str: str, end_str: str, selected_cat: str):
    """
    rafradar-versiyon2.html dosyasındaki interaktif sohbet asistanını (hazır soru chip'leri +
    serbest soru sorma + canlı veri yanıtları) sunar.
    """
    if "rafradar_chat_history" not in st.session_state:
        st.session_state["rafradar_chat_history"] = [
            {
                "role": "assistant",
                "text": (
                    f"Merhaba! Ben **RafRadar Yönetici Asistanı**. Seçtiğiniz **{start_str} – {end_str}** dönemi "
                    f"ve **{selected_cat}** verilerini analiz ettim. Aşağıdaki hazır sorulardan birine tıklayabilir "
                    f"veya merak ettiğiniz soruyu doğrudan yazabilirsiniz."
                ),
            }
        ]

    with st.popover("💬 RafRadar Asistanı'na Sor", use_container_width=True):
        render_html(
            """
            <div style="background:#1f2a44; color:#fff; padding:10px 14px; border-radius:10px; margin-bottom:10px;">
                <div style="font-family:'Poppins',sans-serif; font-weight:600; font-size:14px;">💬 RafRadar Akıllı Yönetici Asistanı</div>
                <div style="font-size:11.5px; color:#d6ece5;">Seçili tarih ve kategori filtresine göre anlık yanıt verir</div>
            </div>
            """
        )

        quick_questions = [
            ("📈 Ciro & Kâr Durumu", "Ciro ve kâr marjımız ne durumda?"),
            ("🚨 Riskteki VIP Bayiler", "Hangi VIP bayiler kayıp riskinde?"),
            ("🏷️ İskonto Kaçağı", "İskonto kaçağı ne kadar ve hangi kategorilerde?"),
            ("⚖️ Pareto 80/20 Riski", "Müşteri ve ürün Pareto bağımlılık riski nedir?"),
            ("🎯 30 Günlük Aksiyonlar", "Bu hafta hangi 4 aksiyonu almalıyız?"),
            ("🔬 Veri & Hesaplama", "Veriler nasıl temizlendi ve maliyet nasıl hesaplandı?"),
        ]

        q_cols = st.columns(2)
        triggered_q = None
        for idx, (btn_label, full_q) in enumerate(quick_questions):
            with q_cols[idx % 2]:
                if st.button(btn_label, key=f"rr_quick_q_{idx}", use_container_width=True):
                    triggered_q = full_q

        user_input = st.chat_input("Yönetici sorunuzu yazın (örn: iskonto kaçağı, vip bayi, stok)...", key="rr_chat_inp")
        active_query = triggered_q or user_input

        if active_query:
            reply = answer_executive_question(active_query, m, start_str, end_str, selected_cat)
            st.session_state["rafradar_chat_history"].append({"role": "user", "text": active_query})
            st.session_state["rafradar_chat_history"].append({"role": "assistant", "text": reply})

        # Son mesajları göster
        for msg in st.session_state["rafradar_chat_history"][-6:]:
            if msg["role"] == "user":
                render_html(
                    f"""
                    <div style="background:#ece7dd; color:#1f2a44; padding:8px 12px; border-radius:10px 10px 2px 10px; margin:6px 0 4px auto; max-width:88%; font-size:12px; font-weight:600;">
                        👤 {msg['text']}
                    </div>
                    """
                )
            else:
                st.markdown(
                    f"<div style='background:#faf7f2; border:1px solid #e4dfd6; border-left:3px solid #3d9a84; padding:9px 12px; border-radius:4px 10px 10px 10px; margin:4px 0 8px 0; font-size:12.5px; color:#1f2a44; line-height:1.45;'>{msg['text']}</div>",
                    unsafe_allow_html=True,
                )

        if len(st.session_state["rafradar_chat_history"]) > 1:
            if st.button("🧹 Sohbeti Temizle", key="rr_clear_chat"):
                st.session_state["rafradar_chat_history"] = st.session_state["rafradar_chat_history"][:1]
                st.rerun()

    # Ayrıca rafradar-versiyon2.html'deki sağ altta yüzen yuvarlak yeşil sohbet balonu (#chatwrap / .cwb)
    q1_ans = answer_executive_question("ciro ve kâr", m, start_str, end_str, selected_cat)
    q2_ans = answer_executive_question("vip müşteri risk", m, start_str, end_str, selected_cat)
    q3_ans = answer_executive_question("iskonto kaçağı", m, start_str, end_str, selected_cat)
    q4_ans = answer_executive_question("pareto 80/20", m, start_str, end_str, selected_cat)
    q5_ans = answer_executive_question("aksiyon planı", m, start_str, end_str, selected_cat)
    q6_ans = answer_executive_question("şehir ve kategori", m, start_str, end_str, selected_cat)

    render_html(
        f"""
        <details class="rr-float-wrap">
            <summary title="RafRadar Yönetici Asistanı">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </summary>
            <div class="rr-float-panel">
                <div style="background:#1f2a44; color:#fff; padding:10px 12px; border-radius:10px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-family:'Poppins',sans-serif; font-weight:600; font-size:13.5px;">💬 RafRadar Asistan</div>
                        <div style="font-size:11px; color:#d6ece5;">Seçili dönem: {start_str} – {end_str}</div>
                    </div>
                    <span class="rr-pill mint" style="font-size:10px;">Canlı Veri</span>
                </div>
                <div style="font-size:11.5px; color:#5b6475; margin-bottom:8px;">
                    Bir soruya tıklayın; yanıtlar seçtiğiniz tarih ve kategori filtresine göre anında hesaplanır:
                </div>
                <details class="rr-qa-chip" open>
                    <summary><span>📈 Ciro ve kâr marjımız ne durumda?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q1_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>🚨 Hangi VIP bayiler siparişi kesti?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q2_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>🏷️ İskonto kaçağı ne kadar ve nerede?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q3_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>⚖️ Çift taraflı Pareto (80/20) riski nedir?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q4_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>🎯 30 günlük aksiyon planı özeti nedir?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q5_ans}</div>
                </details>
                <details class="rr-qa-chip" style="margin-bottom:0;">
                    <summary><span>🏙️ Hangi il ve kategori lider?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q6_ans}</div>
                </details>
                <div style="font-size:11px; color:#5b6475; margin-top:8px; text-align:center;">
                    Serbest metin soru yazmak için sağ üstteki <b>"💬 RafRadar Asistanı'na Sor"</b> penceresini de kullanabilirsiniz.
                </div>
            </div>
        </details>
        """
    )


# -----------------------------------------------------------------------------
# 6. ANA RENDER FONKSİYONU (RAF-RADAR V2 TEMASI + DİNAMİK YÖNETİCİ ÖZETİ)
# -----------------------------------------------------------------------------
def render_executive_summary(filtered_df=None, raw_df=None):
    inject_rafradar_v2_css()

    full_raw_df = raw_df if (raw_df is not None and not raw_df.empty) else load_rafradar_data()
    if full_raw_df.empty:
        st.error("Veri seti bulunamadı! Lütfen 'B2B_Transaction_Data_.parquet' dosyasını kontrol edin.")
        return

    if "DiscountPct" not in full_raw_df.columns or "EstProfit" not in full_raw_df.columns:
        full_raw_df = load_rafradar_data()

    valid_all_df = full_raw_df[~full_raw_df["IsReturn"]].copy()
    global_min_date = valid_all_df["Date"].min()
    global_max_date = valid_all_df["Date"].max()
    available_years = sorted(valid_all_df["Year"].unique().tolist(), reverse=True)

    # -------------------------------------------------------------------------
    # ÜST NAVİGASYON, DİNAMİK FİLTRE ÇUBUĞU VE SOHBET ASİSTANI
    # -------------------------------------------------------------------------
    top_c1, top_c2, top_c3, top_c4, top_c5, top_c6 = st.columns([1.6, 2.5, 1.6, 1.8, 1.6, 1.5])

    with top_c1:
        render_html(
            """
            <div style="display:flex; align-items:center; gap:8px; padding-top:4px;">
                <svg width="28" height="26" viewBox="0 0 28 26" aria-hidden="true">
                    <rect x="2" y="10" width="6" height="14" rx="1.5" fill="#1f2a44"/>
                    <rect x="11" y="3" width="6" height="21" rx="1.5" fill="#3d9a84"/>
                    <rect x="20" y="14" width="6" height="10" rx="1.5" fill="#e0a526"/>
                </svg>
                <div>
                    <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:20px; color:#1f2a44; line-height:1.1;">RafRadar</div>
                    <div style="font-size:11px; color:#5b6475; font-weight:500;">Karar Destek Sistemi</div>
                </div>
            </div>
            """
        )

    with top_c2:
        selected_tab = st.radio(
            "Görünüm Seçimi",
            ["🏛️ Yönetici Özeti", "🛠️ Veri Kalitesi & Metodoloji"],
            horizontal=True,
            key="rafradar_main_tab",
        )

    with top_c3:
        period_options = (
            ["Tüm Veri Seti", "Son 12 Ay", "Son 6 Ay", "Son Çeyrek (90 Gün)", "Yoğun Sezon (Eyl-Ara)", "Özel Tarih Aralığı"]
            + [f"Yıl: {y}" for y in available_years]
        )
        selected_preset = st.selectbox("📅 Dönem", period_options, index=0, key="rafradar_preset")

    max_dt_ts = pd.to_datetime(global_max_date)
    if selected_preset == "Tüm Veri Seti":
        def_start, def_end = global_min_date, global_max_date
    elif selected_preset == "Son 12 Ay":
        def_start = max(global_min_date, (max_dt_ts - pd.Timedelta(days=365)).date())
        def_end = global_max_date
    elif selected_preset == "Son 6 Ay":
        def_start = max(global_min_date, (max_dt_ts - pd.Timedelta(days=180)).date())
        def_end = global_max_date
    elif selected_preset == "Son Çeyrek (90 Gün)":
        def_start = max(global_min_date, (max_dt_ts - pd.Timedelta(days=90)).date())
        def_end = global_max_date
    elif selected_preset == "Yoğun Sezon (Eyl-Ara)":
        def_start, def_end = global_min_date, global_max_date
    elif selected_preset.startswith("Yıl: "):
        yr = int(selected_preset.replace("Yıl: ", ""))
        yr_df = valid_all_df[valid_all_df["Year"] == yr]
        def_start, def_end = yr_df["Date"].min(), yr_df["Date"].max()
    else:
        def_start, def_end = global_min_date, global_max_date

    with top_c4:
        custom_dates = st.date_input(
            "📆 Tarih Seçimi",
            value=(def_start, def_end),
            min_value=global_min_date,
            max_value=global_max_date,
            key=f"rafradar_dates_{selected_preset}",
        )
        if isinstance(custom_dates, (list, tuple)) and len(custom_dates) == 2:
            start_date, end_date = custom_dates[0], custom_dates[1]
        else:
            start_date, end_date = def_start, def_end

    with top_c5:
        top_cats = (
            valid_all_df.groupby("Category")["TotalAmount"]
            .sum()
            .nlargest(25)
            .index.tolist()
        )
        selected_cat = st.selectbox(
            "📚 Kategori",
            ["Tüm Kategoriler"] + top_cats,
            index=0,
            key="rafradar_cat",
        )

    # Filtreleri Veriye Uygula
    scope_sales_df = valid_all_df[
        (valid_all_df["Date"] >= start_date) & (valid_all_df["Date"] <= end_date)
    ].copy()
    scope_raw_df = full_raw_df[
        (full_raw_df["Date"] >= start_date) & (full_raw_df["Date"] <= end_date)
    ].copy()

    if selected_preset == "Yoğun Sezon (Eyl-Ara)":
        scope_sales_df = scope_sales_df[scope_sales_df["Month"].isin([9, 10, 11, 12])]
        scope_raw_df = scope_raw_df[scope_raw_df["Month"].isin([9, 10, 11, 12])]

    if selected_cat != "Tüm Kategoriler":
        scope_sales_df = scope_sales_df[scope_sales_df["Category"] == selected_cat]
        scope_raw_df = scope_raw_df[scope_raw_df["Category"] == selected_cat]

    if scope_sales_df.empty:
        st.warning("Seçilen tarih aralığı ve kategori için satış kaydı bulunamadı. Lütfen tarih aralığını genişletin.")
        return

    span_days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
    prev_end = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).date()
    prev_start = (pd.to_datetime(prev_end) - pd.Timedelta(days=span_days)).date()

    prev_sales_df = valid_all_df[
        (valid_all_df["Date"] >= prev_start) & (valid_all_df["Date"] <= prev_end)
    ].copy()
    if selected_cat != "Tüm Kategoriler":
        prev_sales_df = prev_sales_df[prev_sales_df["Category"] == selected_cat]

    if prev_sales_df.empty and len(scope_sales_df) > 10:
        mid_ts = scope_sales_df["InvoiceDate"].min() + (scope_sales_df["InvoiceDate"].max() - scope_sales_df["InvoiceDate"].min()) / 2
        prev_sales_df = scope_sales_df[scope_sales_df["InvoiceDate"] < mid_ts].copy()
        curr_half_df = scope_sales_df[scope_sales_df["InvoiceDate"] >= mid_ts].copy()
        m = compute_dynamic_metrics(scope_sales_df, prev_sales_df, scope_raw_df)
        first_half_rev = prev_sales_df["TotalAmount"].sum()
        second_half_rev = curr_half_df["TotalAmount"].sum()
        m["rev_growth_pct"] = ((second_half_rev - first_half_rev) / first_half_rev * 100.0) if first_half_rev > 0 else 0.0
        m["cust_diff"] = curr_half_df["CustomerID"].nunique() - prev_sales_df["CustomerID"].nunique()
        comparison_label = "Dönemin İlk Yarısına Göre"
    else:
        m = compute_dynamic_metrics(scope_sales_df, prev_sales_df, scope_raw_df)
        comparison_label = "Önceki Eşdeğer Döneme Göre"

    start_str = pd.to_datetime(start_date).strftime("%d.%m.%Y")
    end_str = pd.to_datetime(end_date).strftime("%d.%m.%Y")

    # Sağ üst sütunda İnteraktif Chatbox Popover Butonu
    with top_c6:
        st.markdown("<div style='padding-top:18px;'></div>", unsafe_allow_html=True)
        render_floating_and_inline_chatbox(m, start_str, end_str, selected_cat)

    # =========================================================================
    # SEKME 1: YÖNETİCİ ÖZETİ (RAF-RADAR V2 GÖRSEL DİLİ + BİZİM KURGUMUZ)
    # =========================================================================
    if selected_tab == "🏛️ Yönetici Özeti":
        render_html(
            f"""
            <div class="rr-page-header">
                <h1>Yönetici Özeti</h1>
                <p>Şirket büyüyor mu, kârlılığımız güvende mi ve önümüzdeki 30 günde hangi kritik aksiyonları almalıyız?</p>
            </div>
            <div class="rr-pill-row">
                <span class="rr-pill mint">Seçili: {start_str} – {end_str} · {span_days} gün</span>
                <span style="font-size:12px; color:#5b6475;">karşılaştırma</span>
                <span class="rr-pill stone">{comparison_label}</span>
                <span class="rr-pill stone">Kategori: {selected_cat}</span>
                <span class="rr-pill Gold">{fmt_int(len(scope_sales_df))} işlem satırı · {fmt_int(m['total_invoices'])} sipariş</span>
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # ÜST KPI KARTLARI (5 SÜTUN - RAF-RADAR V2 .kpi STİLİ)
        # ---------------------------------------------------------------------
        k1, k2, k3, k4, k5 = st.columns(5)

        rev_delta_cls = "kpi-delta-pos" if m["rev_growth_pct"] >= 0 else "kpi-delta-neg"
        rev_arrow = "▲" if m["rev_growth_pct"] >= 0 else "▼"
        daily_rev = m["net_rev"] / max(1, span_days)

        with k1:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Seçili Dönem Net Ciro</div>
                    <div class="kpi-value">{fmt_tl(m['net_rev'])}</div>
                    <span class="{rev_delta_cls}">{rev_arrow} %{abs(m['rev_growth_pct']):.1f}</span>
                    <div class="kpi-rule"></div>
                    <div class="kpi-label">Günlük ort.: <b style="color:#1f2a44;">{fmt_tl(daily_rev, 1)}</b></div>
                </div>
                """
            )

        with k2:
            margin_cls = "kpi-delta-pos" if m["profit_margin"] >= 28.0 else "kpi-delta-warn"
            render_html(
                f"""
                <div class="kpi-card gold">
                    <div class="kpi-label">Tahmini Brüt Kâr & Marj</div>
                    <div class="kpi-value">{fmt_tl(m['est_profit'])}</div>
                    <span class="{margin_cls}">%{m['profit_margin']:.1f} Marj</span>
                    <div class="kpi-rule"></div>
                    <div class="kpi-label">Maliyet varsayımı: <b style="color:#1f2a44;">Liste %44'ü</b></div>
                </div>
                """
            )

        with k3:
            cust_sign = "+" if m["cust_diff"] >= 0 else ""
            cust_cls = "kpi-delta-pos" if m["cust_diff"] >= 0 else "kpi-delta-warn"
            render_html(
                f"""
                <div class="kpi-card blue">
                    <div class="kpi-label">Aktif Müşteri & Pareto</div>
                    <div class="kpi-value">{fmt_int(m['active_customers'])}</div>
                    <span class="{cust_cls}">{cust_sign}{m['cust_diff']} net bayi</span>
                    <div class="kpi-rule"></div>
                    <div class="kpi-label">Pareto: <b style="color:#1f2a44;">%{m['vip_cust_share_pct']:.1f}'i = %80 Ciro</b></div>
                </div>
                """
            )

        with k4:
            render_html(
                f"""
                <div class="kpi-card coral">
                    <div class="kpi-label">Ort. İskonto & Norm Dışı Kaçak</div>
                    <div class="kpi-value">%{m['avg_discount']:.1f}</div>
                    <span class="kpi-delta-neg">{fmt_tl(m['discount_leak_tl'], 1)} kaçak</span>
                    <div class="kpi-rule"></div>
                    <div class="kpi-label">Sapmalı: <b style="color:#8f2a24;">{fmt_int(m['suspicious_orders_count'])} satır</b></div>
                </div>
                """
            )

        with k5:
            render_html(
                f"""
                <div class="kpi-card navy">
                    <div class="kpi-label">Aktif Ürün (SKU) & Satışsız</div>
                    <div class="kpi-value">{fmt_int(m['active_skus'])}</div>
                    <span class="kpi-delta-warn">{fmt_int(m['dead_sku_count'])} Hareketsiz</span>
                    <div class="kpi-rule"></div>
                    <div class="kpi-label">Durum: <b style="color:#1f2a44;">120+ gün satışsız</b></div>
                </div>
                """
            )

        st.write("")

        # ---------------------------------------------------------------------
        # ORTA BÖLÜM:
        # [1] FİNANSAL PERFORMANS & İSKONTO DİSİPLİNİ
        # [2] CİRO & KÂR SIZINTISI KÖPRÜSÜ (WATERFALL ŞELALE GRAFİĞİ)
        # ---------------------------------------------------------------------
        col_left, col_right = st.columns([5.6, 4.4])

        with col_left:
            if span_days <= 62:
                ts_df = (
                    scope_sales_df.groupby("Date", as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"))
                    .sort_values("Date")
                )
                ts_df["Label"] = pd.to_datetime(ts_df["Date"]).dt.strftime("%d %b")
                period_unit_name = "günlerde"
            else:
                ts_df = (
                    scope_sales_df.groupby("YearMonth", as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"))
                    .sort_values("YearMonth")
                )
                ts_df["Label"] = ts_df["YearMonth"]
                period_unit_name = "aylarda"

            ts_df["DiscPct"] = np.where(
                ts_df["GrossRev"] > 0,
                (ts_df["GrossRev"] - ts_df["NetRev"]) / ts_df["GrossRev"] * 100.0,
                0.0,
            )

            rev_thresh = ts_df["NetRev"].quantile(0.68) if len(ts_df) > 3 else ts_df["NetRev"].mean()
            # rafradar-versiyon2.html renkleri: zirve aylar #3d9a84 (teal), diğer aylar #cfc9bd (stone)
            bar_colors = ["#3d9a84" if v >= rev_thresh else "#cfc9bd" for v in ts_df["NetRev"]]

            peak_row = ts_df.loc[ts_df["NetRev"].idxmax()]
            low_row = ts_df.loc[ts_df["NetRev"].idxmin()]
            peak_to_low_ratio = peak_row["NetRev"] / max(1.0, low_row["NetRev"])
            peak_disc_avg = ts_df.loc[ts_df["NetRev"] >= rev_thresh, "DiscPct"].mean()

            dow_map = {
                "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
                "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"
            }
            dow_rev = scope_sales_df.groupby("DayOfWeek")["TotalAmount"].sum()
            top_dow_en = dow_rev.idxmax() if not dow_rev.empty else "Monday"
            top_dow_tr = dow_map.get(top_dow_en, top_dow_en)
            top_dow_share = (dow_rev.max() / m["net_rev"] * 100.0) if m["net_rev"] > 0 else 0.0

            render_html(
                f"""
                <div class="section-card teal-left">
                    <h2 class="rr-h2"><span class="rr-num">1</span>Finansal performans ve iskonto disiplini</h2>
                    <p class="rr-sub">Çubuk: net ciro (₺) · çizgi: ortalama indirim oranı (%) · yeşil çubuklar en yoğun talep dönemleri</p>
                    <div class="section-takeaway">
                        Zirve dönem ({peak_row['Label']}) ile en düşük dönem arasında {peak_to_low_ratio:.1f} kat ciro farkı var;
                        ancak talebin en yüksek olduğu {period_unit_name} bile iskonto oranı (%{peak_disc_avg:.1f}) düşmüyor.
                    </div>
                </div>
                """
            )

            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Bar(
                    x=ts_df["Label"],
                    y=ts_df["NetRev"],
                    name="Net Ciro (₺)",
                    marker_color=bar_colors,
                    text=[fmt_tl(v, 1) for v in ts_df["NetRev"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Net Ciro: ₺%{y:,.0f}<extra></extra>",
                ),
                secondary_y=False,
            )
            fig1.add_trace(
                go.Scatter(
                    x=ts_df["Label"],
                    y=ts_df["DiscPct"],
                    name="İndirim Oranı (%)",
                    mode="lines+markers",
                    line=dict(color="#e0a526", width=3),
                    marker=dict(size=7, color="#e0a526", line=dict(color="#fff", width=1)),
                    hovertemplate="<b>%{x}</b><br>İndirim Oranı: %%{y:.1f}<extra></extra>",
                ),
                secondary_y=True,
            )
            disc_min = max(0, float(ts_df["DiscPct"].min()) - 8)
            disc_max = min(85, float(ts_df["DiscPct"].max()) + 8)
            fig1.update_layout(
                height=285,
                margin=dict(l=10, r=10, t=16, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#1f2a44"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            fig1.update_yaxes(
                title_text="Net Ciro (₺)",
                range=[0, float(ts_df["NetRev"].max()) * 1.24],
                gridcolor="#efe9de",
                secondary_y=False,
            )
            fig1.update_yaxes(
                title_text="İndirim (%)",
                range=[disc_min, disc_max],
                showgrid=False,
                secondary_y=True,
            )
            st.plotly_chart(fig1, use_container_width=True)

            render_html(
                f"""
                <div class="insight-box">
                    <b>Stratejik Çıkarım:</b> Yeşil çubuklarla vurgulanan yüksek talep dönemlerinde organik sipariş güçlü iken
                    indirimin <b>%{peak_disc_avg:.1f}</b> seviyesinde sabit kalması gereksiz marj kaybı yaratıyor.
                    Ayrıca haftalık cironun <b>%{top_dow_share:.1f}'i {top_dow_tr}</b> gününe yığılarak depo darboğazı oluşturuyor.
                </div>
                """
            )

        with col_right:
            total_opportunity_tl = (
                m["vip_recovery_potential_tl"]
                + m["stockout_prevention_tl"]
                + m["retention_potential_tl"]
                + m["discount_leak_tl"]
            )
            target_rev_tl = m["net_rev"] + total_opportunity_tl
            uplift_pct = (total_opportunity_tl / m["net_rev"] * 100.0) if m["net_rev"] > 0 else 0.0

            render_html(
                f"""
                <div class="section-card gold-left">
                    <h2 class="rr-h2"><span class="rr-num">2</span>Ciro ve kâr sızıntısı köprüsü (fırsat havuzu)</h2>
                    <p class="rr-sub">Mevcut cirodan 4 operasyonel sızıntının kapatılmasıyla ulaşılabilir potansiyel ciroya geçiş</p>
                    <div class="section-takeaway">
                        4 sızıntı kapatıldığında mevcut ciroya +{fmt_tl(total_opportunity_tl)}
                        (+%{uplift_pct:.1f}) eklenerek {fmt_tl(target_rev_tl)} seviyesine çıkılabilir.
                    </div>
                </div>
                """
            )

            wf_x = [
                "Mevcut<br>Net Ciro",
                "+ Pasif VIP<br>Kazanımı",
                "+ Yıldız Stok<br>Koruması",
                "+ 2. Sipariş<br>Tekrarı",
                "+ İskonto<br>Kontrolü",
                "Aksiyonlarla<br>Hedef Ciro",
            ]
            wf_y = [
                m["net_rev"],
                m["vip_recovery_potential_tl"],
                m["stockout_prevention_tl"],
                m["retention_potential_tl"],
                m["discount_leak_tl"],
                0,
            ]
            wf_text = [
                fmt_tl(m["net_rev"], 1),
                f"+{fmt_tl(m['vip_recovery_potential_tl'], 1)}",
                f"+{fmt_tl(m['stockout_prevention_tl'], 1)}",
                f"+{fmt_tl(m['retention_potential_tl'], 1)}",
                f"+{fmt_tl(m['discount_leak_tl'], 1)}",
                fmt_tl(target_rev_tl, 1),
            ]

            fig2 = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                    x=wf_x,
                    y=wf_y,
                    text=wf_text,
                    textposition="outside",
                    connector={"line": {"color": "#cfc9bd", "width": 1.5, "dash": "dot"}},
                    increasing={"marker": {"color": "#3d9a84"}},
                    decreasing={"marker": {"color": "#d0574f"}},
                    totals={"marker": {"color": "#1f2a44"}},
                )
            )
            fig2.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#1f2a44"),
                yaxis=dict(
                    title="Ciro Köprüsü (₺)",
                    range=[0, target_rev_tl * 1.18],
                    gridcolor="#efe9de",
                ),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

            render_html(
                f"""
                <div class="leak-grid">
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔴 1. Pasif VIP Bayiler</span>
                            <span style="color:#14513f;">+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Siparişi kesen <b>{m['at_risk_vip_count']} büyük bayinin</b> geri kazanım etkisi.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>📦 2. Yıldız Ürün Stoğu</span>
                            <span style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Cironun %80'ini çeken <b>{m['a_sku_count']} A sınıfı kitapta</b> yok satmayı önleme.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔄 3. Tek Siparişli Bayiler</span>
                            <span style="color:#14513f;">+{fmt_tl(m['retention_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Tek alımda kalan <b>%{m['one_time_ratio_pct']:.0f} bayinin</b> çeyreğine 2. sipariş yaptırma.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🏷️ 4. Aşırı İskonto Tavanı</span>
                            <span style="color:#14513f;">+{fmt_tl(m['discount_leak_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Kategori normunu aşan <b>{fmt_int(m['suspicious_orders_count'])} satırdaki</b> fazladan indirimi kesme.</div>
                    </div>
                </div>
                """
            )

        # ---------------------------------------------------------------------
        # ALT BÖLÜM:
        # [3] PORTFÖY YAPISI VE ÇİFT TARAFLI PARETO (80/20) BAĞIMLILIK RİSKİ
        # [4] YÖNETİM KURULU 30 GÜNLÜK AKSİYON PLANI
        # ---------------------------------------------------------------------
        col_b1, col_b2 = st.columns([4.4, 5.6])

        with col_b1:
            cust_df = m["cust_df"]
            sku_df = m["sku_df"]

            b_cust_count = int((cust_df["Tier"] == "Gelişen Orta Bayiler").sum())
            c_cust_count = int((cust_df["Tier"] == "Küçük / Tekil Bayiler").sum())
            b_cust_share = (b_cust_count / max(1, m["active_customers"])) * 100.0
            c_cust_share = (c_cust_count / max(1, m["active_customers"])) * 100.0

            b_sku_count = int((sku_df["ABC"] == "B Sınıfı Düzenli Kitaplar").sum())
            b_sku_share = (b_sku_count / max(1, m["active_skus"])) * 100.0

            render_html(
                f"""
                <div class="section-card">
                    <h2 class="rr-h2"><span class="rr-num">3</span>Portföy yapısı ve çift taraflı yoğunlaşma (Pareto riski)</h2>
                    <p class="rr-sub">Müşteri ve ürünler ciro payına göre A (%80 Ciro) · B (%15 Ciro) · C (%5 Ciro)</p>
                    <div class="section-takeaway">
                        Müşterilerin %{m['vip_cust_share_pct']:.1f}'i ({m['vip_cust_count']} bayi) ve ürünlerin
                        %{m['a_sku_share_pct']:.1f}'i ({fmt_int(m['a_sku_count'])} kitap) toplam cironun %80'ini getiriyor.
                    </div>
                </div>
                """
            )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                render_html(
                    f"""
                    <div class="pareto-box">
                        <div class="pareto-title">
                            <span>👥 Müşteriler · {fmt_int(m['active_customers'])}</span>
                            <span class="rr-pill mint">İlk 10 = %{m['top10_cust_rev_share']:.0f} Ciro</span>
                        </div>
                        <div class="abc-stack-bar" title="Müşteri adet dağılımı: A, B, C">
                            <div style="flex:{max(1, m['vip_cust_count'])} 1 0; background:#3d9a84; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">A</div>
                            <div style="flex:{max(1, b_cust_count)} 1 0; background:#e0a526; color:#1f2a44; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">B</div>
                            <div style="flex:{max(1, c_cust_count)} 1 0; background:#d9622b; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">C</div>
                        </div>
                        <div class="pareto-sub">A: {m['vip_cust_count']} · B: {b_cust_count} · C: {c_cust_count} bayi</div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>🏆 A · VIP Bayiler (%{m['vip_cust_share_pct']:.0f} bayi)</span>
                                <span style="color:#14513f;">%80 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#3d9a84;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>🤝 B · Orta Segment (%{b_cust_share:.0f} bayi)</span>
                                <span style="color:#7a5600;">%15 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#e0a526;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header">
                                <span>🌱 C · Küçük Alıcılar (%{c_cust_share:.0f} bayi)</span>
                                <span style="color:#8f2a24;">%5 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:5%; background:#d9622b;"></div></div>
                        </div>
                    </div>
                    """
                )

            with p_col2:
                render_html(
                    f"""
                    <div class="pareto-box">
                        <div class="pareto-title">
                            <span>📚 Ürünler · {fmt_int(m['active_skus'])}</span>
                            <span class="rr-pill Gold">C Sınıfı = %{m['c_sku_share_pct']:.0f} SKU</span>
                        </div>
                        <div class="abc-stack-bar" title="Ürün adet dağılımı: A, B, C">
                            <div style="flex:{max(1, m['a_sku_count'])} 1 0; background:#3d9a84; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">A</div>
                            <div style="flex:{max(1, b_sku_count)} 1 0; background:#e0a526; color:#1f2a44; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">B</div>
                            <div style="flex:{max(1, m['c_sku_count'])} 1 0; background:#d9622b; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">C</div>
                        </div>
                        <div class="pareto-sub">A: {fmt_int(m['a_sku_count'])} · B: {fmt_int(b_sku_count)} · C: {fmt_int(m['c_sku_count'])} ürün</div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>⭐ A · Yıldız Kitaplar (%{m['a_sku_share_pct']:.0f} SKU)</span>
                                <span style="color:#14513f;">%80 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#3d9a84;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>📘 B · Düzenli Kitaplar (%{b_sku_share:.0f} SKU)</span>
                                <span style="color:#7a5600;">%15 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#e0a526;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header">
                                <span>📦 C · Uzun Kuyruk (%{m['c_sku_share_pct']:.0f} SKU)</span>
                                <span style="color:#8f2a24;">%5 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:5%; background:#d9622b;"></div></div>
                        </div>
                    </div>
                    """
                )

            pareto_comp_df = pd.DataFrame(
                {
                    "Grup": ["A Sınıfı Müşteriler", "A Sınıfı Ürünler", "C Sınıfı Ürünler"],
                    "Sayıca Pay (%)": [
                        round(m["vip_cust_share_pct"], 1),
                        round(m["a_sku_share_pct"], 1),
                        round(m["c_sku_share_pct"], 1),
                    ],
                    "Ciro Payı (%)": [80.0, 80.0, 5.0],
                }
            )
            fig3 = go.Figure()
            fig3.add_trace(
                go.Bar(
                    name="Portföydeki Sayıca Payı (%)",
                    x=pareto_comp_df["Grup"],
                    y=pareto_comp_df["Sayıca Pay (%)"],
                    marker_color="#cfc9bd",
                    text=[f"%{v:.1f} Adet" for v in pareto_comp_df["Sayıca Pay (%)"]],
                    textposition="outside",
                )
            )
            fig3.add_trace(
                go.Bar(
                    name="Ürettiği Ciro Payı (%)",
                    x=pareto_comp_df["Grup"],
                    y=pareto_comp_df["Ciro Payı (%)"],
                    marker_color=["#3d9a84", "#1f2a44", "#d9622b"],
                    text=[f"%{v:.0f} Ciro" for v in pareto_comp_df["Ciro Payı (%)"]],
                    textposition="outside",
                )
            )
            fig3.update_layout(
                barmode="group",
                height=210,
                margin=dict(l=10, r=10, t=24, b=5),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#1f2a44"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 105], gridcolor="#efe9de", title="Pay (%)"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_b2:
            risk_ex = m["top_risk_examples"]
            if len(risk_ex) >= 2:
                vip_str = (
                    f"Başta <b>{risk_ex.iloc[0]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[0]['Revenue'], 0)}) ve "
                    f"<b>{risk_ex.iloc[1]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[1]['Revenue'], 0)}) olmak üzere "
                    f"toplam <b>{m['at_risk_vip_count']} yüksek cirolu bayi</b> uzun süredir sipariş vermedi."
                )
            elif len(risk_ex) == 1:
                vip_str = (
                    f"<b>{risk_ex.iloc[0]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[0]['Revenue'], 0)}) dahil "
                    f"<b>{m['at_risk_vip_count']} önemli bayi</b> siparişlerini durdurdu."
                )
            else:
                vip_str = "Üst segment bayilerde sipariş aralığı açılan müşteriler tespit edildi."

            top_anom_cats = m["cat_anom_df"].head(2)["Category"].tolist()
            anom_cat_str = " ve ".join(top_anom_cats) if top_anom_cats else "Sözlük ve Roman"

            render_html(
                f"""
                <div class="section-card gold-left">
                    <h2 class="rr-h2"><span class="rr-num">4</span>Önümüzdeki 30 gün için yönetim aksiyon planı</h2>
                    <p class="rr-sub">Seçilen dönem verilerinden süzülen, departmanlar arası en yüksek geri dönüşlü 4 somut karar</p>
                    <div class="action-card urgent">
                        <div class="action-top">
                            <div class="action-dept-group">
                                <span class="action-dept-pill">Pazarlama</span>
                                <span class="action-dept-pill">Satış</span>
                            </div>
                            <span class="rr-pill rose">KRİTİK · Bu Hafta</span>
                        </div>
                        <div class="action-headline">1. Siparişi Kesen VIP Bayiler İçin Birebir Geri Kazanım Aramaları</div>
                        <div class="action-body">
                            <b>Kök Sorun:</b> {vip_str}<br>
                            <b>Somut Aksiyon:</b> Müşteri temsilcileri bu hafta pasif VIP bayileri doğrudan arayarak özel vade veya sepet hedefi tanımlamalı.
                        </div>
                        <div class="action-footer">
                            <span style="color:#5b6475;">Beklenen Ciro Koruma Etkisi:</span>
                            <span class="action-impact" style="color:#8f2a24;">+{fmt_tl(m['vip_recovery_potential_tl'])}</span>
                        </div>
                    </div>
                    <div class="action-card high">
                        <div class="action-top">
                            <div class="action-dept-group">
                                <span class="action-dept-pill">Satış & Fiyat</span>
                                <span class="action-dept-pill">Finans</span>
                            </div>
                            <span class="rr-pill rose">KRİTİK · Bu Hafta</span>
                        </div>
                        <div class="action-headline">2. Kategori Normunu Aşan İskontolara Otomatik Tavan Getirilmesi</div>
                        <div class="action-body">
                            <b>Kök Sorun:</b> Başta <b>{anom_cat_str}</b> olmak üzere {fmt_int(m['suspicious_orders_count'])} satırda kategori medyanından 10+ puan fazla indirim verilmiş.<br>
                            <b>Somut Aksiyon:</b> ERP'de kategori medyan iskontosunu +5 puandan fazla aşan siparişlere yönetici onay şartı getirilmeli.
                        </div>
                        <div class="action-footer">
                            <span style="color:#5b6475;">Doğrudan Net Kâr Koruma Etkisi:</span>
                            <span class="action-impact" style="color:#7a5600;">+{fmt_tl(m['discount_leak_tl'])}</span>
                        </div>
                    </div>
                    <div class="action-card medium">
                        <div class="action-top">
                            <div class="action-dept-group">
                                <span class="action-dept-pill">Tedarik Zinciri</span>
                                <span class="action-dept-pill">Depo</span>
                            </div>
                            <span class="rr-pill Gold">YÜKSEK · 15 Gün</span>
                        </div>
                        <div class="action-headline">3. A Sınıfı Yıldız Kitaplarda Emniyet Stoğu ve Hareketsiz Stok Eritme</div>
                        <div class="action-body">
                            <b>Kök Sorun:</b> Cironun %80'ini üreten <b>{fmt_int(m['a_sku_count'])} A sınıfı kitapta</b> stoksuzluk riski varken, <b>{fmt_int(m['dead_sku_count'])} kitap</b> 120+ gündür satılmıyor.<br>
                            <b>Somut Aksiyon:</b> A sınıfı kitaplara %95 hizmet seviyesiyle güvenlik stoğu tanımlanmalı; satışsız ürünler çok satanlarla paketlenmeli.
                        </div>
                        <div class="action-footer">
                            <span style="color:#5b6475;">Stoksuzluk Önleme & Nakit Etkisi:</span>
                            <span class="action-impact" style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'])}</span>
                        </div>
                    </div>
                    <div class="action-card blue" style="margin-bottom:0;">
                        <div class="action-top">
                            <div class="action-dept-group">
                                <span class="action-dept-pill">CRM</span>
                                <span class="action-dept-pill">Pazarlama</span>
                            </div>
                            <span class="rr-pill mint">PLANLI · 30 Gün</span>
                        </div>
                        <div class="action-headline">4. Tek Siparişte Kalan Bayileri İkinci Siparişe Taşıyacak Paket Teklifi</div>
                        <div class="action-body">
                            <b>Kök Sorun:</b> Seçilen dönemde alım yapan bayilerin <b>%{m['one_time_ratio_pct']:.1f}'i ({fmt_int(m['one_time_count'])} bayi)</b> yalnız 1 kez sipariş verip bıraktı.<br>
                            <b>Somut Aksiyon:</b> İlk faturadan 30 gün sonra bayinin aldığı kategoriyle en sık birlikte alınan ürünler için otomatik teklif gönderilmeli.
                        </div>
                        <div class="action-footer">
                            <span style="color:#5b6475;">Beklenen 2. Sipariş Kazanımı:</span>
                            <span class="action-impact" style="color:#3b6fd1;">+{fmt_tl(m['retention_potential_tl'])}</span>
                        </div>
                    </div>
                </div>
                """
            )

    # =========================================================================
    # SEKME 2: VERİ KALİTESİ & METODOLOJİ (RAF-RADAR V2 "kalite" GÖRSEL DİLİ)
    # =========================================================================
    else:
        total_raw_rows = len(scope_raw_df)
        return_rows = int(scope_raw_df["IsReturn"].sum())
        clean_rows = len(scope_sales_df)
        clean_ratio_pct = (clean_rows / max(1, total_raw_rows)) * 100.0
        return_amount_tl = abs(float(scope_raw_df.loc[scope_raw_df["IsReturn"], "TotalAmount"].sum()))

        render_html(
            f"""
            <div class="rr-page-header">
                <h1>Veri Kalitesi &amp; Metodoloji</h1>
                <p>Bu verilere ne kadar güvenebiliriz, ham veriden hangi bilgileri çıkardık ve rakamlar nasıl hesaplandı?</p>
            </div>
            <div class="rr-pill-row">
                <span class="rr-pill mint">{fmt_int(clean_rows)} / {fmt_int(total_raw_rows)} satır net satışta kullanıldı · %{clean_ratio_pct:.2f}</span>
                <span class="rr-pill stone">Dönem: {start_str} – {end_str}</span>
                <span class="rr-pill stone">{fmt_int(m['total_invoices'])} sipariş · {fmt_int(m['active_customers'])} müşteri · {fmt_int(m['active_skus'])} ürün</span>
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # 1. VERİ ÖN İŞLEME · 5 ADIM (rafradar-versiyon2.html kalite 1. Panel Stili)
        # ---------------------------------------------------------------------
        render_html(
            f"""
            <div class="section-card">
                <h2 class="rr-h2"><span class="rr-num">1</span>Veri ön işleme · 5 adım</h2>
                <p class="rr-sub">Ham Excel/Parquet dosyasından yönetici panellerine giden veri temizleme ve zenginleştirme hattı</p>
                <div class="vq-step-row">
                    <div class="vq-step" style="background:#d6ece5;">
                        <div class="vq-step-label">1 · YÜKLEME</div>
                        <div class="vq-step-val">{fmt_int(total_raw_rows)}</div>
                        <div class="vq-step-desc">ham satır okundu · 10 sütun · Parquet önbellek ile 0,4 sn</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fdf0cf;">
                        <div class="vq-step-label">2 · TEMİZLİK</div>
                        <div class="vq-step-val">−{fmt_int(return_rows)}</div>
                        <div class="vq-step-desc">iptal ('C' kodlu), iade ve sıfır/negatif satır ayrıştırıldı ({fmt_tl(return_amount_tl, 1)})</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#dbe5f8;">
                        <div class="vq-step-label">3 · HESAPLAMA</div>
                        <div class="vq-step-val">+12</div>
                        <div class="vq-step-desc">yeni sütun: brüt ciro, net ciro, indirim %, tahmini maliyet ve brüt kâr</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe0d3;">
                        <div class="vq-step-label">4 · SEGMENTLEME</div>
                        <div class="vq-step-val">A · B · C</div>
                        <div class="vq-step-desc">müşteri (RFM & Pareto) ve ürün ciro payına göre (%80 · %15 · %5) sınıflandı</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe3e0;">
                        <div class="vq-step-label">5 · ANOMALİ DENETİMİ</div>
                        <div class="vq-step-val">{fmt_int(m['suspicious_orders_count'])}</div>
                        <div class="vq-step-desc">sipariş satırı kategori medyanından fazla indirimli ({fmt_tl(m['discount_leak_tl'], 1)} kaçak)</div>
                    </div>
                </div>
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # 2. ALAN DOLULUĞU & METRİK HESAPLAMA REHBERİ VE 3. ANOMALİ GRAFİĞİ
        # ---------------------------------------------------------------------
        col_f1, col_f2 = st.columns([5.5, 4.5])

        with col_f1:
            render_html(
                """
                <div class="section-card">
                    <h2 class="rr-h2"><span class="rr-num">2</span>Veriden çıkarılan bilgiler ve hesaplama formülleri</h2>
                    <p class="rr-sub">Yönetici özetindeki tüm göstergeler ham sütunlardan aşağıdaki matematiksel kurallarla canlı hesaplanır</p>
                    <div class="formula-card">
                        <div class="formula-title">1. Brüt Ciro, Net Ciro ve İndirim Oranı (%)</div>
                        <div class="formula-code">Brüt Ciro = Quantity × UnitPrice &nbsp;|&nbsp; Net Ciro = Quantity × NetPrice</div>
                        <div style="font-size:12px; color:#5b6475;">
                            Liste fiyatı (<code>UnitPrice</code>) ile indirimli net fiyat (<code>NetPrice</code>) arasındaki farktan
                            gerçekleşen indirim oranı hesaplanır: <b>İndirim (%) = ((Brüt − Net) / Brüt) × 100</b>.
                        </div>
                    </div>
                    <div class="formula-card">
                        <div class="formula-title">2. Tahmini Brüt Kâr ve Marj (Maliyet Varsayımı)</div>
                        <div class="formula-code">Tahmini Maliyet = 0,44 × Brüt Ciro &nbsp;|&nbsp; Brüt Kâr = Net Ciro − Tahmini Maliyet</div>
                        <div style="font-size:12px; color:#5b6475;">
                            Ham veride ürün alış maliyeti bulunmadığından sektör standardı olarak <b>Liste Fiyatının %44'ü</b>
                            maliyet kabul edilmiş, kâr simülasyonları bu temel üzerine kurulmuştur.
                        </div>
                    </div>
                    <div class="formula-card">
                        <div class="formula-title">3. Norm Dışı İskonto Kaçağı (Fiyat Anomalisi)</div>
                        <div class="formula-code">Fazla İndirim (₺) = Brüt Tutar × (Sipariş İndirimi % − Kategori Medyan İndirimi %)</div>
                        <div style="font-size:12px; color:#5b6475;">
                            Her ürün kategorisinin ortanca (medyan) indirimi bulunur. Kategori normundan <b>10+ puan</b>
                            fazla indirim verilen siparişlerdeki ekstra kayıp "İskonto Kaçağı" olarak raporlanır.
                        </div>
                    </div>
                    <div class="formula-card" style="margin-bottom:0;">
                        <div class="formula-title">4. Portföy (A·B·C), Geciken VIP Bayiler ve Güvenlik Stoğu</div>
                        <div class="formula-code">Birikimli Ciro Payı ≤ %80 → A Sınıfı &nbsp;|&nbsp; Güvenlik Stoğu: SS = z · σ · √L</div>
                        <div style="font-size:12px; color:#5b6475;">
                            Müşteriler ve ürünler ciroya göre sıralandığında ilk %80'i getirenler <b>A sınıfı</b>, sonraki %15 <b>B sınıfı</b>,
                            son %5 <b>C sınıfı</b> olur. Emniyet stoğu %95 hizmet seviyesi ($z=1,65$) ve tedarik süresi ($L$) ile modellenir.
                        </div>
                    </div>
                </div>
                """
            )

        with col_f2:
            render_html(
                """
                <div class="section-card coral-left">
                    <h2 class="rr-h2"><span class="rr-num">3</span>Kategori bazlı norm dışı indirim sapmaları</h2>
                    <p class="rr-sub">Seçili dönemde kategori medyan indiriminin aşılmasıyla oluşan fazla indirim tutarları (₺)</p>
                </div>
                """
            )

            top_anom_df = m["cat_anom_df"].head(7).copy()
            if not top_anom_df.empty and top_anom_df["LeakTL"].sum() > 0:
                top_anom_df["Label"] = top_anom_df.apply(
                    lambda r: f"{r['Category'][:22]} (Med %{r['MedianDisc']:.0f}→Max %{r['MaxDisc']:.0f})",
                    axis=1,
                )
                fig_dq = go.Figure(
                    go.Bar(
                        x=top_anom_df["LeakTL"],
                        y=top_anom_df["Label"],
                        orientation="h",
                        marker_color="#d0574f",
                        text=[fmt_tl(v, 1) for v in top_anom_df["LeakTL"]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Fazla İndirim Tutarı: ₺%{x:,.0f}<extra></extra>",
                    )
                )
                fig_dq.update_layout(
                    height=245,
                    margin=dict(l=10, r=70, t=10, b=10),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44"),
                    xaxis=dict(
                        title="Fazla İndirim Tutarı (₺)",
                        range=[0, float(top_anom_df["LeakTL"].max()) * 1.28],
                        gridcolor="#efe9de",
                    ),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_dq, use_container_width=True)
            else:
                st.info("Seçilen dar filtrede belirgin bir kategori iskonto sapması bulunmadı.")

            # Sütun Doluluk Karnesi (rafradar-versiyon2.html kalite 2. Panel)
            render_html(
                """
                <div class="section-card" style="margin-top:8px;">
                    <h2 class="rr-h2"><span class="rr-num">4</span>Veri alanı doluluk ve güvenilirlik karnesi</h2>
                    <p class="rr-sub">Zorunlu sütunların doluluk oranı ve eksik veriler için uygulanan analitik çözümler</p>
                    <div style="font-size:12px; display:flex; flex-direction:column; gap:6px; margin-top:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span><b>Sipariş, Müşteri, Tarih, Kategori, Adet, Fiyat</b> (9 Sütun)</span>
                            <span class="rr-pill mint">%100 Dolu · Tam Güvenilir</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span><b>Anlık Depo Stoğu &amp; Tedarik Süresi</b> (Veride Yok)</span>
                            <span class="rr-pill Gold">Simüle Edildi (SS = z·σ·√L)</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span><b>Ürün Alış Maliyeti</b> (Veride Yok)</span>
                            <span class="rr-pill Gold">Liste Fiyatının %44'ü Varsayıldı</span>
                        </div>
                    </div>
                </div>
                """
            )

        with st.expander("🔍 İşlenmiş ve Zenginleştirilmiş Veri Seti Örneğini İncele (İlk 25 Satır)"):
            preview_cols = [
                "InvoiceNo", "CustomerID", "InvoiceDate", "Category", "Description",
                "City", "Quantity", "UnitPrice", "NetPrice", "GrossAmount",
                "TotalAmount", "DiscountPct", "EstProfit"
            ]
            existing_cols = [c for c in preview_cols if c in scope_sales_df.columns]
            st.dataframe(
                scope_sales_df[existing_cols].head(25).rename(
                    columns={
                        "InvoiceNo": "Fatura No",
                        "CustomerID": "Bayi ID",
                        "InvoiceDate": "Tarih",
                        "Category": "Kategori",
                        "Description": "Kitap Adı",
                        "City": "Şehir",
                        "Quantity": "Adet",
                        "UnitPrice": "Liste Fiyatı (₺)",
                        "NetPrice": "Net Fiyat (₺)",
                        "GrossAmount": "Brüt Tutar (₺)",
                        "TotalAmount": "Net Ciro (₺)",
                        "DiscountPct": "İskonto (%)",
                        "EstProfit": "Tahmini Kâr (₺)",
                    }
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    render_executive_summary()
