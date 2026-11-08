import os
import datetime
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI (SOL MENÜ AÇIK - RAFRADAR V2 STANDART YAPI)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RafRadar Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
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
    tüm türetilmiş değişkenleri hazırlar.
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

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    if "NetPrice" in df.columns:
        df["NetPrice"] = pd.to_numeric(df["NetPrice"], errors="coerce").fillna(df["UnitPrice"])
    else:
        df["NetPrice"] = df["UnitPrice"]

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

    df["IsReturn"] = (
        df["InvoiceNo_Str"].str.upper().str.startswith("C")
        | df["InvoiceNo_Str"].str.upper().str.contains("IADE")
        | (df["Quantity"] <= 0)
        | (df["UnitPrice"] <= 0)
    )

    df["GrossAmount"] = df["Quantity"] * df["UnitPrice"]
    df["TotalAmount"] = df["Quantity"] * df["NetPrice"]
    df["DiscountAmount"] = np.maximum(0.0, df["GrossAmount"] - df["TotalAmount"])
    df["DiscountPct"] = np.where(
        df["GrossAmount"] > 0,
        np.clip((df["DiscountAmount"] / df["GrossAmount"]) * 100.0, 0.0, 95.0),
        0.0,
    )

    # Tahmini Alış Maliyeti = Liste Fiyatının %44'ü
    df["EstCost"] = df["GrossAmount"] * 0.44
    df["EstProfit"] = df["TotalAmount"] - df["EstCost"]

    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Quarter"] = df["InvoiceDate"].dt.year.astype(str) + "-Ç" + df["InvoiceDate"].dt.quarter.astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    return df


# -----------------------------------------------------------------------------
# 3. FORMATLAMA FONKSİYONLARI VE RAFRADAR V2 CSS STİLİ
# -----------------------------------------------------------------------------
MONTH_TR_SHORT = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]


def fmt_date_tr(d: datetime.date) -> str:
    """Tarihi '1 Kas 2020' formatına çevirir."""
    return f"{d.day} {MONTH_TR_SHORT[d.month - 1]} {d.year}"


def fmt_range_tr(d1: datetime.date, d2: datetime.date) -> str:
    """Tarih aralığını '1 Kas – 14 Ara 2020' formatına çevirir."""
    if d1.year == d2.year:
        return f"{d1.day} {MONTH_TR_SHORT[d1.month - 1]} – {d2.day} {MONTH_TR_SHORT[d2.month - 1]} {d2.year}"
    return f"{fmt_date_tr(d1)} – {fmt_date_tr(d2)}"


def fmt_tl(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "0 ₺"
    sign = "-" if val < 0 else ""
    v = abs(float(val))
    if v >= 1_000_000:
        return f"{sign}{v / 1_000_000:,.{decimals}f} M ₺".replace(",", "X").replace(".", ",").replace("X", ".")
    if v >= 1_000:
        return f"{sign}{v / 1_000:,.1f} bin ₺".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}{v:,.0f} ₺".replace(",", ".")


def fmt_int(val: float) -> str:
    if pd.isna(val):
        return "0"
    return f"{int(round(val)):,}".replace(",", ".")


def chg_badge_html(pct: float, invert: bool = False) -> str:
    """rafradar-versiyon2.html'deki .chg (.pos / .neg / .mute) rozetini üretir."""
    if pd.isna(pct) or abs(pct) < 0.15:
        return '<span class="chg mute">%0,0</span>'
    is_good = (pct < 0) if invert else (pct > 0)
    cls = "pos" if is_good else "neg"
    sign = "+" if pct > 0 else "−"
    val_str = f"{abs(pct):.1f}".replace(".", ",")
    return f'<span class="chg {cls}">{sign}%{val_str}</span>'


def inject_rafradar_v2_css():
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
        }
        .stApp {
            background-color: #f6f3ee !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            color: #1f2a44;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
        }
        /* Sol Kenar Çubuğu (#sidecol / #side) */
        section[data-testid="stSidebar"] {
            background-color: #faf7f2 !important;
            border-right: 1px solid #e4dfd6 !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem !important;
        }
        /* Üst Sayfa Başlığı (head() fonksiyonu karşılığı) */
        .rr-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .rr-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 28px;
            color: #1f2a44;
            margin: 0;
            line-height: 1.2;
        }
        .rr-header p {
            margin: 4px 0 0 0;
            font-size: 14px;
            font-style: italic;
            color: #5b6475;
        }
        /* Seçili & Karşılaştırma Dönemi Hapları (pills() karşılığı) */
        .rr-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
            margin: 8px 0 14px 0;
        }
        .pill {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 999px;
            white-space: nowrap;
        }
        /* KPI Kartları (kcard() birebir karşılığı) */
        .card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 18px 20px;
            box-sizing: border-box;
        }
        .kpi {
            border-top: 4px solid #3d9a84;
            padding: 14px 16px;
            height: 100%;
        }
        .kl {
            font-size: 12px;
            color: #5b6475;
        }
        .kv {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 22px;
            color: #1f2a44;
            margin: 2px 0;
        }
        .kv2 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #1f2a44;
            margin: 2px 0;
        }
        .chg {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .pos  { background: #d6ece5; color: #14513f; }
        .neg  { background: #fbe3e0; color: #8f2a24; }
        .mute { background: #f0ebe2; color: #5b6475; }
        .neu  { background: #e4e7ee; color: #2b3550; }
        .rule {
            border-top: 1px dashed #d9d3c7;
            margin: 8px 0 6px 0;
        }
        /* Bilgi Notu Bandı (note() karşılığı) */
        .rr-note {
            font-size: 12px;
            color: #5b6475;
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 8px 14px;
            margin: 12px 0 14px 0;
        }
        /* Bölüm Kartları (.h, .sub, .num) */
        .h {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 15px;
            color: #1f2a44;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .sub {
            font-size: 12px;
            color: #5b6475;
            margin: 4px 0 0 0;
        }
        .num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #1f2a44;
            color: #ffffff;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
            flex-shrink: 0;
        }
        .section-takeaway {
            font-size: 14px;
            font-weight: 600;
            color: #1f2a44;
            margin: 8px 0 6px 0;
            line-height: 1.4;
        }
        .insight-box {
            background-color: #faf7f2;
            border: 1px solid #e4dfd6;
            border-left: 4px solid #3d9a84;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12px;
            color: #1f2a44;
            margin-top: 8px;
            margin-bottom: 14px;
            line-height: 1.5;
        }
        /* Sızıntı / Fırsat Mini Kartları */
        .leak-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
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
        /* Portföy / Pareto Kutuları */
        .pareto-box {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .abc-stack-bar {
            display: flex;
            height: 28px;
            border-radius: 8px;
            overflow: hidden;
            margin: 6px 0 8px 0;
        }
        .tier-row { margin-bottom: 8px; }
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
        .tier-bar-fill { height: 100%; border-radius: 999px; }

        /* 30 Günlük Aksiyon Kartları */
        .action-card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-top: 4px solid #3d9a84;
            border-radius: 10px;
            padding: 12px 14px;
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

        /* Veri Kalitesi 5 Adım Kutuları */
        .vq-step-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 12px 0 4px 0;
            align-items: stretch;
        }
        .vq-step {
            flex: 1 1 135px;
            min-width: 0;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .vq-step-label {
            font-size: 11px;
            font-weight: 600;
            color: #5b6475;
        }
        .vq-step-val {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 22px;
            color: #1f2a44;
            margin: 4px 0;
        }
        .vq-step-desc {
            font-size: 12px;
            color: #1f2a44;
            line-height: 1.35;
        }
        .vq-arrow {
            align-self: center;
            color: #5b6475;
            font-size: 18px;
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

        /* Sağ Alt Yüzen Sohbet Balonu (#chatwrap / .cwb) */
        .rr-float-wrap {
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 9999;
        }
        .rr-float-wrap > summary {
            list-style: none;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #3d9a84;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(31, 42, 68, 0.25);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 0;
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
            bottom: 66px;
            width: 360px;
            max-width: calc(100vw - 24px);
            max-height: calc(100vh - 100px);
            overflow-y: auto;
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            box-shadow: 0 10px 28px rgba(31, 42, 68, 0.25);
            padding: 14px;
            color: #1f2a44;
        }
        .rr-qa-chip {
            border: 1px solid #cfc9bd;
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


def render_kcard(
    label: str,
    main_val: str,
    main_chg_html: str,
    sub_label: str,
    sub_val: str,
    sub_chg_html: str,
    top_color: str = "#3d9a84",
):
    """rafradar-versiyon2.html içindeki kcard() fonksiyonunun birebir karşılığı."""
    render_html(
        f"""
        <div class="card kpi" style="border-top-color:{top_color};">
            <div class="kl">{label}</div>
            <div style="display:flex; align-items:baseline; justify-content:space-between; gap:6px; flex-wrap:wrap;">
                <div class="kv">{main_val}</div>
                {main_chg_html}
            </div>
            <div class="rule"></div>
            <div style="display:flex; align-items:baseline; justify-content:space-between; gap:6px; flex-wrap:wrap;">
                <span class="kl">{sub_label}</span>
                <div>
                    <span class="kv2">{sub_val}</span>
                    {sub_chg_html}
                </div>
            </div>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 4. SOL MENÜ (SIDEBAR): RAFRADAR V2 STANDART FİLTRE VE NAVİGASYON YAPISI
# -----------------------------------------------------------------------------
def render_rafradar_sidebar(valid_all_df: pd.DataFrame, show_page_nav: bool = True) -> dict:
    """
    rafradar-versiyon2.html'deki sol sütunu (#sidecol / #side) birebir oluşturur:
    - RafRadar Logo & Başlık
    - Sayfa / Bölüm Seçimi (Yönetici Özeti, Veri Kalitesi)
    - Seçili Dönem (fa – fb, varsayılan: 1 Kas – 14 Ara 2020)
    - Karşılaştırma Dönemi (fca – fcb, varsayılan: 1 Eyl – 31 Eki 2020)
    - [Önceki dönem] ve [Sıfırla] butonları
    - Kategori Filtresi
    """
    global_min = valid_all_df["Date"].min()
    global_max = valid_all_df["Date"].max()

    # rafradar-versiyon2.html varsayılan tarihleri:
    # F = { a: '2020-11-01', b: '2020-12-14', ca: '2020-09-01', cb: '2020-10-31' }
    def_a = max(global_min, min(global_max, datetime.date(2020, 11, 1)))
    def_b = max(global_min, min(global_max, datetime.date(2020, 12, 14)))
    def_ca = max(global_min, min(global_max, datetime.date(2020, 9, 1)))
    def_cb = max(global_min, min(global_max, datetime.date(2020, 10, 31)))

    if "rr_fa" not in st.session_state:
        st.session_state["rr_fa"] = def_a
        st.session_state["rr_fb"] = def_b
        st.session_state["rr_fca"] = def_ca
        st.session_state["rr_fcb"] = def_cb

    with st.sidebar:
        render_html(
            """
            <div style="display:flex; align-items:center; gap:9px; padding-bottom:12px; border-bottom:1px solid #e4dfd6; margin-bottom:12px;">
                <svg width="28" height="26" viewBox="0 0 28 26" aria-hidden="true">
                    <rect x="2" y="10" width="6" height="14" fill="#1f2a44"/>
                    <rect x="11" y="3" width="6" height="21" fill="#3d9a84"/>
                    <rect x="20" y="14" width="6" height="10" fill="#e0a526"/>
                </svg>
                <div>
                    <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:20px; color:#1f2a44; line-height:1.1;">RafRadar</div>
                    <div style="font-size:11px; color:#5b6475;">B2B Karar Destek Sistemi</div>
                </div>
            </div>
            """
        )

        if show_page_nav:
            selected_page = st.radio(
                "Bölüm Seçimi",
                ["Yönetici Özeti", "Veri Kalitesi"],
                index=0,
                key="rr_side_page",
            )
            render_html("<div style='border-top:1px solid #e4dfd6; margin:10px 0;'></div>")
        else:
            selected_page = "Yönetici Özeti"

        # Hızlı Dönem Kısayolu + Seçili Dönem (fa, fb)
        render_html(
            "<div style='font-family:Poppins,sans-serif; font-size:12.5px; font-weight:600; color:#1f2a44; margin-bottom:4px;'>📅 Seçili dönem</div>"
        )
        preset = st.selectbox(
            "Hızlı Dönem Seç",
            [
                "Özel / Varsayılan (1 Kas – 14 Ara)",
                "Tüm Yıl (1 Oca – 31 Ara 2020)",
                "Son Çeyrek (Eki – Ara 2020)",
                "Okul Sezonu (Eyl – Kas 2020)",
                "Yaz Dönemi (Haz – Ağu 2020)",
                "İlk Yarıyıl (Oca – Haz 2020)",
            ],
            index=0,
            label_visibility="collapsed",
            key="rr_preset_sel",
        )

        if preset == "Tüm Yıl (1 Oca – 31 Ara 2020)":
            st.session_state["rr_fa"] = global_min
            st.session_state["rr_fb"] = global_max
        elif preset == "Son Çeyrek (Eki – Ara 2020)":
            st.session_state["rr_fa"] = max(global_min, datetime.date(2020, 10, 1))
            st.session_state["rr_fb"] = global_max
            st.session_state["rr_fca"] = max(global_min, datetime.date(2020, 7, 1))
            st.session_state["rr_fcb"] = max(global_min, datetime.date(2020, 9, 30))
        elif preset == "Okul Sezonu (Eyl – Kas 2020)":
            st.session_state["rr_fa"] = max(global_min, datetime.date(2020, 9, 1))
            st.session_state["rr_fb"] = max(global_min, datetime.date(2020, 11, 30))
            st.session_state["rr_fca"] = max(global_min, datetime.date(2020, 6, 1))
            st.session_state["rr_fcb"] = max(global_min, datetime.date(2020, 8, 31))
        elif preset == "Yaz Dönemi (Haz – Ağu 2020)":
            st.session_state["rr_fa"] = max(global_min, datetime.date(2020, 6, 1))
            st.session_state["rr_fb"] = max(global_min, datetime.date(2020, 8, 31))
            st.session_state["rr_fca"] = max(global_min, datetime.date(2020, 3, 1))
            st.session_state["rr_fcb"] = max(global_min, datetime.date(2020, 5, 31))
        elif preset == "İlk Yarıyıl (Oca – Haz 2020)":
            st.session_state["rr_fa"] = global_min
            st.session_state["rr_fb"] = max(global_min, datetime.date(2020, 6, 30))
            st.session_state["rr_fca"] = max(global_min, datetime.date(2020, 7, 1))
            st.session_state["rr_fcb"] = global_max

        sc1, sc2 = st.columns(2)
        with sc1:
            fa = st.date_input(
                "Başlangıç",
                value=st.session_state["rr_fa"],
                min_value=global_min,
                max_value=global_max,
                key="rr_fa_inp",
            )
        with sc2:
            fb = st.date_input(
                "Bitiş",
                value=st.session_state["rr_fb"],
                min_value=global_min,
                max_value=global_max,
                key="rr_fb_inp",
            )
        if fa > fb:
            fa, fb = fb, fa
        st.session_state["rr_fa"], st.session_state["rr_fb"] = fa, fb

        # Karşılaştırma Dönemi (fca, fcb)
        render_html(
            "<div style='font-family:Poppins,sans-serif; font-size:12.5px; font-weight:600; color:#1f2a44; margin:10px 0 4px 0;'>⚖️ Karşılaştırma dönemi</div>"
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            fca = st.date_input(
                "Karş. Başlangıç",
                value=st.session_state["rr_fca"],
                min_value=global_min,
                max_value=global_max,
                key="rr_fca_inp",
            )
        with cc2:
            fcb = st.date_input(
                "Karş. Bitiş",
                value=st.session_state["rr_fcb"],
                min_value=global_min,
                max_value=global_max,
                key="rr_fcb_inp",
            )
        if fca > fcb:
            fca, fcb = fcb, fca
        st.session_state["rr_fca"], st.session_state["rr_fcb"] = fca, fcb

        # rafradar-versiyon2.html'deki [Önceki dönem] (#prevper) ve [Sıfırla] (#reset) butonları
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Önceki dönem", use_container_width=True, key="rr_btn_prevper"):
                days_len = max(1, (fb - fa).days + 1)
                new_cb = max(global_min, fa - datetime.timedelta(days=1))
                new_ca = max(global_min, new_cb - datetime.timedelta(days=days_len - 1))
                st.session_state["rr_fca"] = new_ca
                st.session_state["rr_fcb"] = new_cb
                st.rerun()
        with b_col2:
            if st.button("Sıfırla", use_container_width=True, key="rr_btn_reset"):
                st.session_state["rr_fa"] = def_a
                st.session_state["rr_fb"] = def_b
                st.session_state["rr_fca"] = def_ca
                st.session_state["rr_fcb"] = def_cb
                st.rerun()

        render_html("<div style='border-top:1px solid #e4dfd6; margin:10px 0;'></div>")

        # Kategori Filtresi
        top_cats = (
            valid_all_df.groupby("Category")["TotalAmount"]
            .sum()
            .nlargest(25)
            .index.tolist()
        )
        selected_cat = st.selectbox(
            "📚 Kategori Filtresi",
            ["Tüm Kategoriler"] + top_cats,
            index=0,
            key="rr_side_cat",
        )

    return {
        "page": selected_page,
        "fa": st.session_state["rr_fa"],
        "fb": st.session_state["rr_fb"],
        "fca": st.session_state["rr_fca"],
        "fcb": st.session_state["rr_fcb"],
        "category": selected_cat,
    }


# -----------------------------------------------------------------------------
# 5. DİNAMİK ANALİTİK HESAPLAMA MOTORU (SEÇİLİ vs KARŞILAŞTIRMA DÖNEMİ)
# -----------------------------------------------------------------------------
def compute_dynamic_metrics(sales_df: pd.DataFrame, comp_sales_df: pd.DataFrame, raw_scope_df: pd.DataFrame, s_days: int, c_days: int) -> dict:
    net_rev = sales_df["TotalAmount"].sum()
    gross_rev = sales_df["GrossAmount"].sum()
    est_profit = sales_df["EstProfit"].sum()
    profit_margin = (est_profit / net_rev * 100.0) if net_rev > 0 else 0.0
    avg_discount = ((gross_rev - net_rev) / gross_rev * 100.0) if gross_rev > 0 else 0.0

    comp_net_rev = comp_sales_df["TotalAmount"].sum() if not comp_sales_df.empty else 0.0
    comp_gross_rev = comp_sales_df["GrossAmount"].sum() if not comp_sales_df.empty else 0.0
    comp_profit = comp_sales_df["EstProfit"].sum() if not comp_sales_df.empty else 0.0
    comp_discount = ((comp_gross_rev - comp_net_rev) / comp_gross_rev * 100.0) if comp_gross_rev > 0 else 0.0

    def rel_pct(a, b):
        if b == 0 or pd.isna(b):
            return 0.0
        return ((a - b) / abs(b)) * 100.0

    # Toplam ve Günlük Ortalama Değişimler (rafradar-versiyon2.html S ve C mantığı)
    d_net_rev = net_rev / max(1, s_days)
    d_comp_rev = comp_net_rev / max(1, c_days)

    d_profit = est_profit / max(1, s_days)
    d_comp_profit = comp_profit / max(1, c_days)

    active_customers = sales_df["CustomerID"].nunique()
    comp_customers = comp_sales_df["CustomerID"].nunique() if not comp_sales_df.empty else 0
    d_cust = sales_df.groupby("Date")["CustomerID"].nunique().mean() if not sales_df.empty else 0.0
    d_comp_cust = comp_sales_df.groupby("Date")["CustomerID"].nunique().mean() if not comp_sales_df.empty else 0.0

    active_skus = sales_df["Description"].nunique()
    comp_skus = comp_sales_df["Description"].nunique() if not comp_sales_df.empty else 0
    total_invoices = sales_df["InvoiceNo"].nunique()
    comp_invoices = comp_sales_df["InvoiceNo"].nunique() if not comp_sales_df.empty else 0
    d_inv = total_invoices / max(1, s_days)
    d_comp_inv = comp_invoices / max(1, c_days)

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

    # Müşteri Pareto (80/20) ve Kayıp Riski Analizi
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

    recency_threshold = max(14, int(s_days * 0.30))
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
    dead_threshold = 120 if s_days >= 150 else max(21, int(s_days * 0.5))
    dead_sku_count = int((sku_df["DaysSinceSale"] >= dead_threshold).sum())

    a_class_rev = float(sku_df.loc[sku_df["ABC"] == "A Sınıfı Yıldız Kitaplar", "Revenue"].sum())
    stockout_prevention_tl = a_class_rev * 0.045

    city_rev = sales_df.groupby("City")["TotalAmount"].sum().sort_values(ascending=False)
    top_city = city_rev.index[0] if not city_rev.empty else "İstanbul"
    top_city_share = (city_rev.iloc[0] / net_rev * 100.0) if (not city_rev.empty and net_rev > 0) else 0.0

    cat_rev = sales_df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False)
    top_cat_name = cat_rev.index[0] if not cat_rev.empty else "Genel"
    top_cat_share = (cat_rev.iloc[0] / net_rev * 100.0) if (not cat_rev.empty and net_rev > 0) else 0.0

    return {
        "net_rev": net_rev,
        "d_net_rev": d_net_rev,
        "rev_chg": rel_pct(net_rev, comp_net_rev),
        "d_rev_chg": rel_pct(d_net_rev, d_comp_rev),
        "gross_rev": gross_rev,
        "est_profit": est_profit,
        "d_profit": d_profit,
        "profit_chg": rel_pct(est_profit, comp_profit),
        "d_profit_chg": rel_pct(d_profit, d_comp_profit),
        "profit_margin": profit_margin,
        "avg_discount": avg_discount,
        "disc_diff_pts": avg_discount - comp_discount,
        "active_customers": active_customers,
        "cust_chg": rel_pct(active_customers, comp_customers),
        "d_cust": d_cust,
        "d_cust_chg": rel_pct(d_cust, d_comp_cust),
        "active_skus": active_skus,
        "sku_chg": rel_pct(active_skus, comp_skus),
        "total_invoices": total_invoices,
        "inv_chg": rel_pct(total_invoices, comp_invoices),
        "d_inv": d_inv,
        "d_inv_chg": rel_pct(d_inv, d_comp_inv),
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
        "dead_threshold": dead_threshold,
        "stockout_prevention_tl": stockout_prevention_tl,
        "top_city": top_city,
        "top_city_share": top_city_share,
        "top_cat_name": top_cat_name,
        "top_cat_share": top_cat_share,
    }


# -----------------------------------------------------------------------------
# 6. İNTERAKTİF RAF-RADAR SOHBET ASİSTANI (SAĞ ALT YÜZEN CHATBOX + POPOVER)
# -----------------------------------------------------------------------------
def answer_executive_question(query: str, m: dict, sel_lbl: str, comp_lbl: str, selected_cat: str) -> str:
    q = query.lower().strip()
    total_opp = (
        m["vip_recovery_potential_tl"]
        + m["stockout_prevention_tl"]
        + m["retention_potential_tl"]
        + m["discount_leak_tl"]
    )
    target_rev = m["net_rev"] + total_opp

    if any(k in q for k in ["ciro", "büyü", "kâr", "kar", "marj", "finans", "durum"]):
        dir_txt = f"%{abs(m['d_rev_chg']):.1f} artış" if m["d_rev_chg"] >= 0 else f"%{abs(m['d_rev_chg']):.1f} düşüş"
        return (
            f"📊 **Finansal Özet ({sel_lbl}):** Seçili dönemde **{fmt_int(m['total_invoices'])} sipariş** "
            f"üzerinden **{fmt_tl(m['net_rev'])} net ciro** (günlük ort. **{fmt_tl(m['d_net_rev'], 1)}**, karşılaştırma dönemine göre {dir_txt}) üretildi. "
            f"**Tahmini brüt kârımız {fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)** seviyesindedir."
        )

    if any(k in q for k in ["iskonto", "indirim", "kaçak", "anomali", "şüpheli", "fiyat"]):
        top_cats = ", ".join(m["cat_anom_df"].head(3)["Category"].tolist()) if not m["cat_anom_df"].empty else "Sözlük ve Roman"
        return (
            f"🏷️ **İndirim & Kaçak Denetimi:** Seçili dönemde ortalama indirim oranı **%{m['avg_discount']:.1f}**'dir. "
            f"**{fmt_int(m['suspicious_orders_count'])} satırda** kategori medyanının 10+ puan üzerinde indirim verilerek "
            f"**{fmt_tl(m['discount_leak_tl'])}** fazla indirim (kâr kaçağı) oluşmuştur. En çok sapma olan kategoriler: **{top_cats}**."
        )

    if any(k in q for k in ["vip", "müşteri", "bayi", "kayıp", "churn", "risk", "pasif"]):
        risk_ex = m["top_risk_examples"]
        ex_names = ", ".join([f"{r['CustomerID']} ({fmt_tl(r['Revenue'], 0)})" for _, r in risk_ex.iterrows()]) if not risk_ex.empty else "üst segment bayiler"
        return (
            f"🚨 **Müşteri & Kayıp Riski Durumu:** Seçili dönemde **{fmt_int(m['active_customers'])} aktif müşterinin** "
            f"**%{m['vip_cust_share_pct']:.1f}'i ({m['vip_cust_count']} A sınıfı müşteri)** cironun %80'ini üretiyor. "
            f"Başta **{ex_names}** olmak üzere **{m['at_risk_vip_count']} önemli bayi** uzun süredir sipariş vermedi "
            f"(geri kazanım potansiyeli: **+{fmt_tl(m['vip_recovery_potential_tl'])}**)."
        )

    if any(k in q for k in ["pareto", "80/20", "yoğunlaşma", "bağımlılık", "portföy", "abc"]):
        return (
            f"⚖️ **Portföy & Pareto (80/20) Yapısı:** Cironun %80'i müşterilerin **%{m['vip_cust_share_pct']:.1f}'ine "
            f"({m['vip_cust_count']} müşteri)** ve ürünlerin **%{m['a_sku_share_pct']:.1f}'ine ({fmt_int(m['a_sku_count'])} A sınıfı ürün)** "
            f"bağımlıdır. İlk 10 müşteri tek başına cironun **%{m['top10_cust_rev_share']:.1f}**'ini oluşturmaktadır."
        )

    if any(k in q for k in ["stok", "tedarik", "ölü", "hareketsiz", "kitap", "ürün", "sku"]):
        return (
            f"📦 **Stok ve Ürün Portföyü:** Seçili dönemde **{fmt_int(m['active_skus'])} farklı ürün** satıldı. "
            f"Cironun %80'ini çeken **{fmt_int(m['a_sku_count'])} A sınıfı üründe** sıfır stoksuzluk (+{fmt_tl(m['stockout_prevention_tl'])} koruma) "
            f"hedeflenmeli; **{m['dead_threshold']}+ gündür satılmayan {fmt_int(m['dead_sku_count'])} ürün** paket kampanyalarla eritilmelidir."
        )

    if any(k in q for k in ["aksiyon", "ne yapmalı", "karar", "plan", "fırsat", "potansiyel", "30 gün"]):
        return (
            f"🎯 **Önümüzdeki 30 Gün İçin 4 Aksiyon Özeti:** Toplam **+{fmt_tl(total_opp)}** ek potansiyel mevcuttur:\n"
            f"1. **Geciken VIP Müşteri Aramaları:** +{fmt_tl(m['vip_recovery_potential_tl'])}\n"
            f"2. **A Sınıfı Yıldız Ürün Stok Koruması:** +{fmt_tl(m['stockout_prevention_tl'])}\n"
            f"3. **Tek Siparişli Müşterilere 2. Sipariş Teklifi:** +{fmt_tl(m['retention_potential_tl'])}\n"
            f"4. **Kategori İndirim Tavanı (+5 Puan Onay Şartı):** +{fmt_tl(m['discount_leak_tl'])}"
        )

    if any(k in q for k in ["şehir", "il", "bölge", "kategori", "en çok"]):
        return (
            f"🏙️ **İl ve Kategori Lideri:** Seçili dönemde en yüksek ciro getiren il **{m['top_city']}** "
            f"(cironun **%{m['top_city_share']:.1f}**'i), lider kategori ise **{m['top_cat_name']}** "
            f"(cironun **%{m['top_cat_share']:.1f}**'i) olmuştur."
        )

    return (
        f"💡 **Dönem Özeti ({sel_lbl}):** Net ciro **{fmt_tl(m['net_rev'])}** (günlük ort. **{fmt_tl(m['d_net_rev'], 1)}**), "
        f"tahmini brüt kâr **{fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)**, aktif müşteri **{fmt_int(m['active_customers'])}** "
        f"ve 4 aksiyondan toplam erişilebilir fırsat **+{fmt_tl(total_opp)}**'dir."
    )


def render_chatbox_widgets(m: dict, sel_lbl: str, comp_lbl: str, selected_cat: str):
    """Hem üst başlık sağında interaktif popover hem de sağ altta yüzen yeşil sohbet balonu (#chatwrap) çizer."""
    if "rafradar_chat_history" not in st.session_state:
        st.session_state["rafradar_chat_history"] = [
            {
                "role": "assistant",
                "text": (
                    f"Merhaba! Seçtiğiniz **{sel_lbl}** dönemi (karşılaştırma: **{comp_lbl}**) verilerini analiz ettim. "
                    f"Hazır sorulardan birine tıklayabilir veya sorunuzu aşağıya yazabilirsiniz."
                ),
            }
        ]

    with st.popover("💬 RafRadar Asistanı", use_container_width=True):
        render_html(
            f"""
            <div style="background:#1f2a44; color:#fff; padding:10px 14px; border-radius:10px; margin-bottom:10px;">
                <div style="font-family:'Poppins',sans-serif; font-weight:600; font-size:14px;">💬 RafRadar Yönetici Asistanı</div>
                <div style="font-size:11px; color:#d6ece5;">Seçili dönem: {sel_lbl}</div>
            </div>
            """
        )

        quick_questions = [
            ("📈 Ciro & Kâr Durumu", "Ciro ve kâr marjımız ne durumda?"),
            ("🚨 Geciken VIP Müşteriler", "Hangi VIP bayiler kayıp riskinde?"),
            ("🏷️ Fazla İndirim Kaçağı", "İskonto kaçağı ne kadar ve hangi kategorilerde?"),
            ("⚖️ Portföy & Pareto Riski", "Müşteri ve ürün Pareto bağımlılık riski nedir?"),
            ("🎯 30 Günlük Aksiyon Planı", "Bu hafta hangi 4 aksiyonu almalıyız?"),
            ("🏙️ Lider İl & Kategori", "Hangi il ve kategori en çok satıyor?"),
        ]

        q_cols = st.columns(2)
        triggered_q = None
        for idx, (btn_label, full_q) in enumerate(quick_questions):
            with q_cols[idx % 2]:
                if st.button(btn_label, key=f"rr_quick_q_{idx}", use_container_width=True):
                    triggered_q = full_q

        user_input = st.chat_input("Sorunuzu yazın (örn: ciro, indirim kaçağı, vip, stok)...", key="rr_chat_inp")
        active_query = triggered_q or user_input

        if active_query:
            reply = answer_executive_question(active_query, m, sel_lbl, comp_lbl, selected_cat)
            st.session_state["rafradar_chat_history"].append({"role": "user", "text": active_query})
            st.session_state["rafradar_chat_history"].append({"role": "assistant", "text": reply})

        for msg in st.session_state["rafradar_chat_history"][-6:]:
            if msg["role"] == "user":
                render_html(
                    f"""
                    <div style="background:#ece7dd; color:#1f2a44; padding:7px 11px; border-radius:10px 10px 2px 10px; margin:5px 0 3px auto; max-width:88%; font-size:12px; font-weight:600;">
                        👤 {msg['text']}
                    </div>
                    """
                )
            else:
                st.markdown(
                    f"<div style='background:#faf7f2; border:1px solid #e4dfd6; border-left:3px solid #3d9a84; padding:8px 11px; border-radius:4px 10px 10px 10px; margin:3px 0 7px 0; font-size:12px; color:#1f2a44; line-height:1.45;'>{msg['text']}</div>",
                    unsafe_allow_html=True,
                )

        if len(st.session_state["rafradar_chat_history"]) > 1:
            if st.button("🧹 Sohbeti Temizle", key="rr_clear_chat"):
                st.session_state["rafradar_chat_history"] = st.session_state["rafradar_chat_history"][:1]
                st.rerun()

    # Sağ altta yüzen yuvarlak yeşil sohbet ikonu (#chatwrap / .cwb)
    q1_ans = answer_executive_question("ciro ve kâr", m, sel_lbl, comp_lbl, selected_cat)
    q2_ans = answer_executive_question("vip müşteri risk", m, sel_lbl, comp_lbl, selected_cat)
    q3_ans = answer_executive_question("iskonto kaçağı", m, sel_lbl, comp_lbl, selected_cat)
    q4_ans = answer_executive_question("pareto 80/20", m, sel_lbl, comp_lbl, selected_cat)
    q5_ans = answer_executive_question("aksiyon planı", m, sel_lbl, comp_lbl, selected_cat)
    q6_ans = answer_executive_question("şehir ve kategori", m, sel_lbl, comp_lbl, selected_cat)

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
                        <div style="font-size:11px; color:#d6ece5;">Seçili: {sel_lbl}</div>
                    </div>
                    <span class="pill" style="background:#d6ece5; color:#14513f;">Canlı</span>
                </div>
                <div style="font-size:11.5px; color:#5b6475; margin-bottom:8px;">
                    Soruya tıklayın; yanıtlar sol menüdeki seçili tarih aralığına göre anında güncellenir:
                </div>
                <details class="rr-qa-chip" open>
                    <summary><span>📈 Ciro ve kâr marjımız ne durumda?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q1_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>🚨 Hangi VIP müşteriler siparişi kesti?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q2_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>🏷️ Fazla indirim kaçağı ne kadar?</span><span>▾</span></summary>
                    <div class="rr-qa-ans">{q3_ans}</div>
                </details>
                <details class="rr-qa-chip">
                    <summary><span>⚖️ Portföy &amp; Pareto (80/20) riski nedir?</span><span>▾</span></summary>
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
            </div>
        </details>
        """
    )


# -----------------------------------------------------------------------------
# 7. ANA RENDER FONKSİYONU (SOL MENÜ + ÜST BAR + YÖNETİCİ ÖZETİ & VERİ KALİTESİ)
# -----------------------------------------------------------------------------
def render_executive_summary(filtered_df=None, raw_df=None, external_filters: dict = None):
    inject_rafradar_v2_css()

    full_raw_df = raw_df if (raw_df is not None and not raw_df.empty) else load_rafradar_data()
    if full_raw_df.empty:
        st.error("Veri seti bulunamadı! Lütfen 'B2B_Transaction_Data_.parquet' dosyasını kontrol edin.")
        return

    if "DiscountPct" not in full_raw_df.columns or "EstProfit" not in full_raw_df.columns:
        full_raw_df = load_rafradar_data()

    valid_all_df = full_raw_df[~full_raw_df["IsReturn"]].copy()

    # Sol kenar çubuğunu (rafradar-versiyon2.html #side) oluştur veya dışarıdan gelen filtreyi kullan
    if external_filters is not None:
        nav = external_filters
    else:
        nav = render_rafradar_sidebar(valid_all_df, show_page_nav=True)

    fa, fb = nav["fa"], nav["fb"]
    fca, fcb = nav["fca"], nav["fcb"]
    selected_cat = nav.get("category", "Tüm Kategoriler")
    selected_page = nav.get("page", "Yönetici Özeti")

    s_days = max(1, (fb - fa).days + 1)
    c_days = max(1, (fcb - fca).days + 1)

    scope_sales_df = valid_all_df[(valid_all_df["Date"] >= fa) & (valid_all_df["Date"] <= fb)].copy()
    comp_sales_df = valid_all_df[(valid_all_df["Date"] >= fca) & (valid_all_df["Date"] <= fcb)].copy()
    scope_raw_df = full_raw_df[(full_raw_df["Date"] >= fa) & (full_raw_df["Date"] <= fb)].copy()

    if selected_cat != "Tüm Kategoriler":
        scope_sales_df = scope_sales_df[scope_sales_df["Category"] == selected_cat]
        comp_sales_df = comp_sales_df[comp_sales_df["Category"] == selected_cat]
        scope_raw_df = scope_raw_df[scope_raw_df["Category"] == selected_cat]

    if scope_sales_df.empty:
        st.warning("Seçilen tarih aralığı ve kategori için satış kaydı bulunamadı. Lütfen sol menüden tarih aralığını genişletin.")
        return

    m = compute_dynamic_metrics(scope_sales_df, comp_sales_df, scope_raw_df, s_days, c_days)
    sel_lbl = fmt_range_tr(fa, fb)
    comp_lbl = fmt_range_tr(fca, fcb)

    # Üst Başlık ve Sağ Üstte Hızlı Görünüm Geçişi + Sohbet Asistanı
    hdr_left, hdr_mid, hdr_right = st.columns([5.2, 3.0, 1.8])
    with hdr_mid:
        active_subview = st.radio(
            "Görünüm",
            ["🏛️ Yönetici Özeti", "🛠️ Veri Kalitesi"],
            index=0 if selected_page != "Veri Kalitesi" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="rr_top_subview",
        )
    with hdr_right:
        render_chatbox_widgets(m, sel_lbl, comp_lbl, selected_cat)

    # =========================================================================
    # GÖRÜNÜM 1: YÖNETİCİ ÖZETİ (rafradar-versiyon2.html Üst Bar + Bizim Kurgumuz)
    # =========================================================================
    if active_subview == "🏛️ Yönetici Özeti":
        with hdr_left:
            render_html(
                """
                <header class="rr-header">
                    <div>
                        <h1>Yönetici Özeti</h1>
                        <p>Seçtiğim dönem yılın neresinde, kârlılığımız güvende mi ve önümüzdeki 30 günde ne yapmalıyız?</p>
                    </div>
                </header>
                """
            )

        # rafradar-versiyon2.html pills() satırı (Her sayfada aynı olan seçili vs karşılaştırma bandı)
        cat_pill_html = (
            f'<span class="pill" style="background:#dbe5f8;color:#1f2a44;">Kategori: {selected_cat}</span>'
            if selected_cat != "Tüm Kategoriler"
            else ""
        )
        render_html(
            f"""
            <section class="rr-pills">
                <span class="pill" style="background:#d6ece5;color:#14513f;">Seçili: {sel_lbl} · {s_days} gün</span>
                <span style="font-size:12px;color:#5b6475;">karşılaştırma</span>
                <span class="pill" style="background:#ece7dd;color:#1f2a44;">{comp_lbl} · {c_days} gün</span>
                {cat_pill_html}
            </section>
            """
        )

        # ---------------------------------------------------------------------
        # ÜST 5'Lİ KPI KARTLARI (rafradar-versiyon2.html kcard() formatı:
        # Üstte Seçili Dönem Toplamı + % Değişim | Kesikli Çizgi | Altta Günlük Ort. + % Değişim)
        # ---------------------------------------------------------------------
        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            render_kcard(
                label="Net ciro",
                main_val=fmt_tl(m["net_rev"]),
                main_chg_html=chg_badge_html(m["rev_chg"]),
                sub_label="Günlük ort.",
                sub_val=fmt_tl(m["d_net_rev"], 1),
                sub_chg_html=chg_badge_html(m["d_rev_chg"]),
                top_color="#3d9a84",
            )

        with k2:
            render_kcard(
                label="Tahmini brüt kâr",
                main_val=fmt_tl(m["est_profit"]),
                main_chg_html=f'<span class="chg pos">%{m["profit_margin"]:.1f} marj</span>',
                sub_label="Günlük ort.",
                sub_val=fmt_tl(m["d_profit"], 1),
                sub_chg_html=chg_badge_html(m["d_profit_chg"]),
                top_color="#e0a526",
            )

        with k3:
            render_kcard(
                label="Aktif müşteri & Pareto",
                main_val=fmt_int(m["active_customers"]),
                main_chg_html=chg_badge_html(m["cust_chg"]),
                sub_label="Pareto payı",
                sub_val=f"%{m['vip_cust_share_pct']:.1f} bayi",
                sub_chg_html='<span class="chg mute">= %80 Ciro</span>',
                top_color="#3b6fd1",
            )

        with k4:
            disc_pts_str = f"{m['disc_diff_pts']:+.1f} puan".replace(".", ",")
            disc_cls = "neg" if m["disc_diff_pts"] > 0.5 else ("pos" if m["disc_diff_pts"] < -0.5 else "mute")
            render_kcard(
                label="Ort. indirim & kaçak",
                main_val=f"%{m['avg_discount']:.1f}".replace(".", ","),
                main_chg_html=f'<span class="chg {disc_cls}">{disc_pts_str}</span>',
                sub_label="Fazla indirim",
                sub_val=fmt_tl(m["discount_leak_tl"], 1),
                sub_chg_html=f'<span class="chg neg">{fmt_int(m["suspicious_orders_count"])} satır</span>',
                top_color="#d0574f",
            )

        with k5:
            render_kcard(
                label="Aktif ürün & satışsız",
                main_val=fmt_int(m["active_skus"]),
                main_chg_html=chg_badge_html(m["sku_chg"]),
                sub_label=f"{m['dead_threshold']}+ gün pasif",
                sub_val=f"{fmt_int(m['dead_sku_count'])} ürün",
                sub_chg_html='<span class="chg neg">eritilmeli</span>',
                top_color="#1f2a44",
            )

        render_html(
            """
            <div class="rr-note">
                Tüm göstergeler, sızıntı köprüsü, Pareto dağılımı ve 30 günlük aksiyon planı sol menüde seçtiğin
                <b>Seçili dönem</b> ve <b>Karşılaştırma dönemi</b> tarihlerine göre canlı hesaplanır.
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # ORTA BÖLÜM:
        # [1] FİNANSAL PERFORMANS & İNDİRİM DİSİPLİNİ
        # [2] CİRO & KÂR SIZINTISI KÖPRÜSÜ (WATERFALL ŞELALE GRAFİĞİ)
        # ---------------------------------------------------------------------
        col_left, col_right = st.columns([5.5, 4.5])

        with col_left:
            if s_days <= 62:
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
                <div class="card" style="border-left:4px solid #3d9a84; margin-bottom:10px;">
                    <h2 class="h"><span class="num">1</span>Finansal performans ve indirim disiplini</h2>
                    <p class="sub">Seçili dönem ({sel_lbl}) · çubuk: net ciro · sarı çizgi: ortalama indirim oranı (%)</p>
                    <div class="section-takeaway">
                        Zirve dönem ({peak_row['Label']}) ile en düşük dönem arasında {peak_to_low_ratio:.1f} kat ciro farkı var;
                        ancak talebin en yüksek olduğu {period_unit_name} bile indirim oranı (%{peak_disc_avg:.1f}) düşmüyor.
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
                    line=dict(color="#e0a526", width=2.8),
                    marker=dict(size=7, color="#e0a526", line=dict(color="#fff", width=1)),
                    hovertemplate="<b>%{x}</b><br>İndirim Oranı: %%{y:.1f}<extra></extra>",
                ),
                secondary_y=True,
            )
            disc_min = max(0, float(ts_df["DiscPct"].min()) - 8)
            disc_max = min(85, float(ts_df["DiscPct"].max()) + 8)
            fig1.update_layout(
                height=280,
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
                    Ayrıca seçili dönemde cironun <b>%{top_dow_share:.1f}'i {top_dow_tr}</b> gününe yığılıyor.
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
                <div class="card" style="border-left:4px solid #e0a526; margin-bottom:10px;">
                    <h2 class="h"><span class="num">2</span>Ciro ve kâr sızıntısı köprüsü (fırsat havuzu)</h2>
                    <p class="sub">Seçili dönem cirosu + 4 operasyonel sızıntının kapatılmasıyla ulaşılabilir potansiyel</p>
                    <div class="section-takeaway">
                        4 sızıntı kapatıldığında seçili dönem cirosuna +{fmt_tl(total_opportunity_tl)}
                        (+%{uplift_pct:.1f}) eklenerek {fmt_tl(target_rev_tl)} seviyesine çıkılabilir.
                    </div>
                </div>
                """
            )

            wf_x = [
                "Seçili<br>Net Ciro",
                "+ Pasif VIP<br>Kazanımı",
                "+ Yıldız Stok<br>Koruması",
                "+ 2. Sipariş<br>Tekrarı",
                "+ İndirim<br>Kontrolü",
                "Aksiyonlarla<br>Potansiyel",
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
                height=235,
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
                            <span>🔴 1. Pasif VIP Müşteriler</span>
                            <span style="color:#14513f;">+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Siparişi geciken <b>{m['at_risk_vip_count']} büyük bayinin</b> geri kazanım etkisi.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>📦 2. Yıldız Ürün Stoğu</span>
                            <span style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Cironun %80'ini çeken <b>{m['a_sku_count']} A sınıfı üründe</b> yok satmayı önleme.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔄 3. Tek Siparişli Bayiler</span>
                            <span style="color:#14513f;">+{fmt_tl(m['retention_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Tek alımda kalan <b>%{m['one_time_ratio_pct']:.0f} müşterinin</b> çeyreğine 2. sipariş yaptırma.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🏷️ 4. Fazla İndirim Tavanı</span>
                            <span style="color:#14513f;">+{fmt_tl(m['discount_leak_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Kategori medyanını aşan <b>{fmt_int(m['suspicious_orders_count'])} satırdaki</b> kaçağı kesme.</div>
                    </div>
                </div>
                """
            )

        # ---------------------------------------------------------------------
        # ALT BÖLÜM:
        # [3] PORTFÖY YAPISI VE ÇİFT TARAFLI YOĞUNLAŞMA (PARETO 80/20)
        # [4] ÖNÜMÜZDEKİ 30 GÜN İÇİN YÖNETİM AKSİYON PLANI
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
                <div class="card" style="margin-bottom:10px;">
                    <h2 class="h"><span class="num">3</span>Portföy yapısı ve çift taraflı yoğunlaşma (Pareto)</h2>
                    <p class="sub">Seçili dönemde müşteri ve ürünler ciro payına göre A (%80) · B (%15) · C (%5)</p>
                    <div class="section-takeaway">
                        Müşterilerin %{m['vip_cust_share_pct']:.1f}'i ({m['vip_cust_count']} müşteri) ve ürünlerin
                        %{m['a_sku_share_pct']:.1f}'i ({fmt_int(m['a_sku_count'])} ürün) cironun %80'ini getiriyor.
                    </div>
                </div>
                """
            )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                render_html(
                    f"""
                    <div class="pareto-box">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:12.5px; font-weight:600;">Müşteriler · {fmt_int(m['active_customers'])}</span>
                            <span class="pill" style="background:#d6ece5; color:#14513f;">İlk 10 = %{m['top10_cust_rev_share']:.0f} Ciro</span>
                        </div>
                        <div class="abc-stack-bar">
                            <div style="flex:{max(1, m['vip_cust_count'])} 1 0; background:#3d9a84; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">A</div>
                            <div style="flex:{max(1, b_cust_count)} 1 0; background:#e0a526; color:#1f2a44; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">B</div>
                            <div style="flex:{max(1, c_cust_count)} 1 0; background:#d9622b; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">C</div>
                        </div>
                        <div style="font-size:11px; color:#5b6475; margin-bottom:8px;">A {m['vip_cust_count']} · B {b_cust_count} · C {c_cust_count} müşteri</div>
                        <div class="tier-row">
                            <div class="tier-header"><span>A · VIP (%{m['vip_cust_share_pct']:.0f} müşteri)</span><span style="color:#14513f;">%80 Ciro</span></div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#3d9a84;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header"><span>B · Orta (%{b_cust_share:.0f} müşteri)</span><span style="color:#7a5600;">%15 Ciro</span></div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#e0a526;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header"><span>C · Küçük (%{c_cust_share:.0f} müşteri)</span><span style="color:#8f2a24;">%5 Ciro</span></div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:5%; background:#d9622b;"></div></div>
                        </div>
                    </div>
                    """
                )

            with p_col2:
                render_html(
                    f"""
                    <div class="pareto-box">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:12.5px; font-weight:600;">Ürünler · {fmt_int(m['active_skus'])}</span>
                            <span class="pill" style="background:#fdf0cf; color:#7a5600;">C = %{m['c_sku_share_pct']:.0f} ürün</span>
                        </div>
                        <div class="abc-stack-bar">
                            <div style="flex:{max(1, m['a_sku_count'])} 1 0; background:#3d9a84; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">A</div>
                            <div style="flex:{max(1, b_sku_count)} 1 0; background:#e0a526; color:#1f2a44; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">B</div>
                            <div style="flex:{max(1, m['c_sku_count'])} 1 0; background:#d9622b; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600;">C</div>
                        </div>
                        <div style="font-size:11px; color:#5b6475; margin-bottom:8px;">A {fmt_int(m['a_sku_count'])} · B {fmt_int(b_sku_count)} · C {fmt_int(m['c_sku_count'])} ürün</div>
                        <div class="tier-row">
                            <div class="tier-header"><span>A · Yıldız (%{m['a_sku_share_pct']:.0f} ürün)</span><span style="color:#14513f;">%80 Ciro</span></div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#3d9a84;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header"><span>B · Düzenli (%{b_sku_share:.0f} ürün)</span><span style="color:#7a5600;">%15 Ciro</span></div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#e0a526;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header"><span>C · Uzun Kuyruk (%{m['c_sku_share_pct']:.0f} ürün)</span><span style="color:#8f2a24;">%5 Ciro</span></div>
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
                    f"toplam <b>{m['at_risk_vip_count']} yüksek cirolu müşteri</b> uzun süredir sipariş vermedi."
                )
            elif len(risk_ex) == 1:
                vip_str = (
                    f"<b>{risk_ex.iloc[0]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[0]['Revenue'], 0)}) dahil "
                    f"<b>{m['at_risk_vip_count']} önemli müşteri</b> siparişlerini durdurdu."
                )
            else:
                vip_str = "Üst segment müşterilerde sipariş aralığı açılan bayiler tespit edildi."

            top_anom_cats = m["cat_anom_df"].head(2)["Category"].tolist()
            anom_cat_str = " ve ".join(top_anom_cats) if top_anom_cats else "Sözlük ve Roman"

            render_html(
                f"""
                <div class="card" style="border-left:4px solid #e0a526;">
                    <h2 class="h"><span class="num">4</span>Önümüzdeki 30 gün için yönetim aksiyon planı</h2>
                    <p class="sub">Seçilen dönem ({sel_lbl}) verilerinden süzülen, en yüksek geri dönüşlü 4 somut karar</p>
                    <div style="margin-top:10px;">
                        <div class="action-card urgent">
                            <div class="action-top">
                                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Pazarlama</span>
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Satış</span>
                                </div>
                                <span class="pill neg">KRİTİK · Bu Hafta</span>
                            </div>
                            <div class="action-headline">1. Siparişi Geciken VIP Müşteriler İçin Birebir Geri Kazanım Aramaları</div>
                            <div class="action-body">
                                <b>Tespit Edilen Kök Sorun:</b> {vip_str}<br>
                                <b>Önerilen Aksiyon:</b> Müşteri temsilcileri bu hafta pasif VIP bayileri doğrudan arayarak özel vade veya sepet hedefi tanımlamalı.
                            </div>
                            <div class="action-footer">
                                <span style="color:#5b6475;">Beklenen Ciro Koruma Etkisi:</span>
                                <span class="action-impact" style="color:#8f2a24;">+{fmt_tl(m['vip_recovery_potential_tl'])}</span>
                            </div>
                        </div>
                        <div class="action-card high">
                            <div class="action-top">
                                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Satış &amp; Fiyat</span>
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Finans</span>
                                </div>
                                <span class="pill neg">KRİTİK · Bu Hafta</span>
                            </div>
                            <div class="action-headline">2. Kategori Normunu Aşan İndirimlere Otomatik Tavan Getirilmesi</div>
                            <div class="action-body">
                                <b>Tespit Edilen Kök Sorun:</b> Başta <b>{anom_cat_str}</b> olmak üzere {fmt_int(m['suspicious_orders_count'])} satırda kategori medyanından 10+ puan fazla indirim verilmiş.<br>
                                <b>Önerilen Aksiyon:</b> ERP'de kategori medyan indirimini +5 puandan fazla aşan siparişlere yönetici onay şartı getirilmeli.
                            </div>
                            <div class="action-footer">
                                <span style="color:#5b6475;">Doğrudan Net Kâr Koruma Etkisi:</span>
                                <span class="action-impact" style="color:#7a5600;">+{fmt_tl(m['discount_leak_tl'])}</span>
                            </div>
                        </div>
                        <div class="action-card medium">
                            <div class="action-top">
                                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Tedarik Zinciri</span>
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Depo</span>
                                </div>
                                <span class="pill" style="background:#fdf0cf;color:#7a5600;">YÜKSEK · 15 Gün</span>
                            </div>
                            <div class="action-headline">3. A Sınıfı Yıldız Ürünlerde Güvenlik Stoğu ve Satışsız Ürün Eritme</div>
                            <div class="action-body">
                                <b>Tespit Edilen Kök Sorun:</b> Cironun %80'ini üreten <b>{fmt_int(m['a_sku_count'])} A sınıfı üründe</b> stoksuzluk riski varken, <b>{fmt_int(m['dead_sku_count'])} ürün</b> {m['dead_threshold']}+ gündür satılmıyor.<br>
                                <b>Önerilen Aksiyon:</b> A sınıfı ürünlere %95 hizmet seviyesiyle güvenlik stoğu tanımlanmalı; satışsız ürünler çok satanlarla paketlenmeli.
                            </div>
                            <div class="action-footer">
                                <span style="color:#5b6475;">Stoksuzluk Önleme &amp; Nakit Etkisi:</span>
                                <span class="action-impact" style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'])}</span>
                            </div>
                        </div>
                        <div class="action-card blue" style="margin-bottom:0;">
                            <div class="action-top">
                                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                    <span class="pill" style="background:#1f2a44;color:#fff;">CRM</span>
                                    <span class="pill" style="background:#1f2a44;color:#fff;">Pazarlama</span>
                                </div>
                                <span class="pill pos">PLANLI · 30 Gün</span>
                            </div>
                            <div class="action-headline">4. Tek Siparişte Kalan Müşterileri İkinci Siparişe Taşıyacak Paket Teklifi</div>
                            <div class="action-body">
                                <b>Tespit Edilen Kök Sorun:</b> Seçili dönemde alım yapan müşterilerin <b>%{m['one_time_ratio_pct']:.1f}'i ({fmt_int(m['one_time_count'])} müşteri)</b> yalnız 1 kez sipariş verdi.<br>
                                <b>Önerilen Aksiyon:</b> İlk faturadan 30 gün sonra müşterinin aldığı kategoriyle en sık birlikte alınan ürünler için otomatik teklif gönderilmeli.
                            </div>
                            <div class="action-footer">
                                <span style="color:#5b6475;">Beklenen 2. Sipariş Kazanımı:</span>
                                <span class="action-impact" style="color:#3b6fd1;">+{fmt_tl(m['retention_potential_tl'])}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """
            )

    # =========================================================================
    # GÖRÜNÜM 2: VERİ KALİTESİ (rafradar-versiyon2.html Üst Bar + Veri Kalitesi)
    # =========================================================================
    else:
        total_raw_rows = len(scope_raw_df)
        return_rows = int(scope_raw_df["IsReturn"].sum())
        clean_rows = len(scope_sales_df)
        clean_ratio_pct = (clean_rows / max(1, total_raw_rows)) * 100.0
        return_amount_tl = abs(float(scope_raw_df.loc[scope_raw_df["IsReturn"], "TotalAmount"].sum()))

        with hdr_left:
            render_html(
                """
                <header class="rr-header">
                    <div>
                        <h1>Veri Kalitesi</h1>
                        <p>Bu verilere ne kadar güvenebiliriz ve rakamlar nasıl hesaplandı?</p>
                    </div>
                </header>
                """
            )

        render_html(
            f"""
            <section class="rr-pills">
                <span class="pill pos">{fmt_int(clean_rows)} / {fmt_int(total_raw_rows)} satır kullanıldı · %{clean_ratio_pct:.2f}</span>
                <span class="pill" style="background:#ece7dd;color:#1f2a44;">Seçili: {sel_lbl} · {s_days} gün</span>
                <span class="pill" style="background:#ece7dd;color:#1f2a44;">{fmt_int(m['total_invoices'])} sipariş · {fmt_int(m['active_customers'])} müşteri</span>
            </section>
            """
        )

        # 1. VERİ ÖN İŞLEME · 5 ADIM (rafradar-versiyon2.html kalite 1. Panel)
        render_html(
            f"""
            <div class="card" style="margin-bottom:14px;">
                <h2 class="h"><span class="num">1</span>Veri ön işleme · 5 adım</h2>
                <p class="sub">Ham dosyadan panellere giden yol · seçili dönem ({sel_lbl})</p>
                <div class="vq-step-row">
                    <div class="vq-step" style="background:#d6ece5;">
                        <div class="vq-step-label">1 · YÜKLEME</div>
                        <div class="vq-step-val">{fmt_int(total_raw_rows)}</div>
                        <div class="vq-step-desc">satır okundu · 10 sütun · Parquet önbellek</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fdf0cf;">
                        <div class="vq-step-label">2 · TEMİZLİK</div>
                        <div class="vq-step-val">−{fmt_int(return_rows)}</div>
                        <div class="vq-step-desc">iptal, iade ve sıfır/negatif satır çıkarıldı ({fmt_tl(return_amount_tl, 1)})</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#dbe5f8;">
                        <div class="vq-step-label">3 · HESAPLAMA</div>
                        <div class="vq-step-val">+12</div>
                        <div class="vq-step-desc">yeni sütun: brüt ciro, net ciro, indirim, tahmini kâr</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe0d3;">
                        <div class="vq-step-label">4 · SEGMENTLEME</div>
                        <div class="vq-step-val">A · B · C</div>
                        <div class="vq-step-desc">müşteri ve ürün, ciro payına göre (%80 · %15 · %5)</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe3e0;">
                        <div class="vq-step-label">5 · ANOMALİ DENETİMİ</div>
                        <div class="vq-step-val">{fmt_int(m['suspicious_orders_count'])}</div>
                        <div class="vq-step-desc">sipariş satırı normdan fazla indirimli ({fmt_tl(m['discount_leak_tl'], 1)})</div>
                    </div>
                </div>
            </div>
            """
        )

        col_f1, col_f2 = st.columns([5.4, 4.6])

        with col_f1:
            render_html(
                """
                <div class="card">
                    <h2 class="h"><span class="num">2</span>Veriden çıkarılan bilgiler ve hesaplama kuralları</h2>
                    <p class="sub">Yönetici özetindeki tüm göstergeler ham sütunlardan aşağıdaki kurallarla canlı hesaplanır</p>
                    <div style="margin-top:10px;">
                        <div class="formula-card">
                            <div class="formula-title">1. Brüt Ciro, Net Ciro ve İndirim Oranı (%)</div>
                            <div class="formula-code">Brüt Ciro = Quantity × UnitPrice &nbsp;|&nbsp; Net Ciro = Quantity × NetPrice</div>
                            <div style="font-size:12px; color:#5b6475;">
                                Liste fiyatı ile net fiyat arasındaki farktan gerçekleşen indirim oranı hesaplanır:
                                <b>İndirim (%) = ((Brüt − Net) / Brüt) × 100</b>.
                            </div>
                        </div>
                        <div class="formula-card">
                            <div class="formula-title">2. Tahmini Brüt Kâr ve Marj (Maliyet Varsayımı)</div>
                            <div class="formula-code">Tahmini Maliyet = 0,44 × Brüt Ciro &nbsp;|&nbsp; Brüt Kâr = Net Ciro − Tahmini Maliyet</div>
                            <div style="font-size:12px; color:#5b6475;">
                                Veride alış maliyeti bulunmadığından sektör standardı olarak <b>Liste Fiyatının %44'ü</b>
                                maliyet kabul edilmiş, kâr hesapları bu temel üzerine kurulmuştur.
                            </div>
                        </div>
                        <div class="formula-card">
                            <div class="formula-title">3. Norm Dışı İndirim Kaçağı (Anomali Taraması)</div>
                            <div class="formula-code">Fazla İndirim (₺) = Brüt Tutar × (Sipariş İndirimi % − Kategori Medyan İndirimi %)</div>
                            <div style="font-size:12px; color:#5b6475;">
                                Her ürün kategorisinin ortanca (medyan) indirimi bulunur. Kategori normundan <b>10+ puan</b>
                                fazla indirim verilen siparişlerdeki ekstra kayıp "Fazla İndirim Kaçağı" olarak raporlanır.
                            </div>
                        </div>
                        <div class="formula-card" style="margin-bottom:0;">
                            <div class="formula-title">4. Portföy (A·B·C) ve Güvenlik Stoğu</div>
                            <div class="formula-code">Birikimli Ciro Payı ≤ %80 → A Sınıfı &nbsp;|&nbsp; Güvenlik Stoğu: SS = z · σ · √L</div>
                            <div style="font-size:12px; color:#5b6475;">
                                Müşteriler ve ürünler ciroya göre sıralandığında ilk %80'i getirenler <b>A sınıfı</b>, sonraki %15 <b>B sınıfı</b>,
                                son %5 <b>C sınıfı</b> olur. Güvenlik stoğu %95 hizmet seviyesi ($z=1,65$) ve tedarik süresi ($L$) ile hesaplanır.
                            </div>
                        </div>
                    </div>
                </div>
                """
            )

        with col_f2:
            # Alan Doluluğu (rafradar-versiyon2.html kalite 2. Panel SVG barlarıyla birebir)
            render_html(
                """
                <div class="card" style="margin-bottom:14px;">
                    <h2 class="h"><span class="num">3</span>Alan doluluğu</h2>
                    <p class="sub">Her alanın yüzde kaçı dolu · sarı olanlar simülasyonla tamamlandı</p>
                    <div style="margin-top:10px;">
                        <div style="display:grid;grid-template-columns:150px 1fr 56px;gap:10px;align-items:center;padding:5px 0;">
                            <span style="font-size:12.5px;">Sipariş &amp; Müşteri No</span>
                            <svg viewBox="0 0 300 10" width="100%"><rect width="300" height="10" rx="5" fill="#efe9de"/><rect width="300" height="10" rx="5" fill="#3d9a84"/></svg>
                            <b style="font-size:12px;text-align:right;">%100</b>
                        </div>
                        <div style="display:grid;grid-template-columns:150px 1fr 56px;gap:10px;align-items:center;padding:5px 0;">
                            <span style="font-size:12.5px;">Tarih &amp; Kategori</span>
                            <svg viewBox="0 0 300 10" width="100%"><rect width="300" height="10" rx="5" fill="#efe9de"/><rect width="300" height="10" rx="5" fill="#3d9a84"/></svg>
                            <b style="font-size:12px;text-align:right;">%100</b>
                        </div>
                        <div style="display:grid;grid-template-columns:150px 1fr 56px;gap:10px;align-items:center;padding:5px 0;">
                            <span style="font-size:12.5px;">Adet, Liste &amp; Net Fiyat</span>
                            <svg viewBox="0 0 300 10" width="100%"><rect width="300" height="10" rx="5" fill="#efe9de"/><rect width="300" height="10" rx="5" fill="#3d9a84"/></svg>
                            <b style="font-size:12px;text-align:right;">%100</b>
                        </div>
                        <div style="display:grid;grid-template-columns:150px 1fr 56px;gap:10px;align-items:center;padding:5px 0;">
                            <span style="font-size:12.5px;">Alış Maliyeti (Liste %44)</span>
                            <svg viewBox="0 0 300 10" width="100%"><rect width="300" height="10" rx="5" fill="#efe9de"/><rect width="300" height="10" rx="5" fill="#e0a526"/></svg>
                            <b style="font-size:11px;text-align:right;color:#7a5600;">Varsayım</b>
                        </div>
                    </div>
                </div>
                """
            )

            render_html(
                """
                <div class="card" style="border-left:4px solid #d0574f;">
                    <h2 class="h"><span class="num">4</span>Kategori bazlı fazla indirim sapmaları</h2>
                    <p class="sub">Seçili dönemde kategori medyan indiriminin aşılmasıyla oluşan fazla indirim tutarları (₺)</p>
                </div>
                """
            )

            top_anom_df = m["cat_anom_df"].head(6).copy()
            if not top_anom_df.empty and top_anom_df["LeakTL"].sum() > 0:
                top_anom_df["Label"] = top_anom_df.apply(
                    lambda r: f"{r['Category'][:20]} (Med %{r['MedianDisc']:.0f}→Max %{r['MaxDisc']:.0f})",
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
                    height=220,
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
                        "CustomerID": "Müşteri No",
                        "InvoiceDate": "Tarih",
                        "Category": "Kategori",
                        "Description": "Ürün Adı",
                        "City": "İl",
                        "Quantity": "Adet",
                        "UnitPrice": "Liste Fiyatı (₺)",
                        "NetPrice": "Net Fiyat (₺)",
                        "GrossAmount": "Brüt Ciro (₺)",
                        "TotalAmount": "Net Ciro (₺)",
                        "DiscountPct": "İndirim (%)",
                        "EstProfit": "Tahmini Kâr (₺)",
                    }
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    render_executive_summary()
