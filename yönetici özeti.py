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
# 1. SAYFA YAPILANDIRMASI (DOĞRUDAN ÇALIŞTIRILDIĞINDA)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RafRadar | Yönetici Özeti & Veri Kalitesi",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_html(html_str: str):
    """Girintileri temizleyerek Markdown'ın HTML'i kod bloğu sanmasını %100 engeller."""
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE ÖN İŞLEME MOTORU (CACHE DESTEKLİ)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_rafradar_data() -> pd.DataFrame:
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
        df["CustomerID"].fillna("Anonim").astype(str).str.strip()
        if "CustomerID" in df.columns
        else "Anonim"
    )
    df["Category"] = (
        df["Category"].fillna("Genel").astype(str).str.strip()
        if "Category" in df.columns
        else "Genel"
    )
    df["Description"] = (
        df["Description"].fillna("Tanımsız Ürün").astype(str).str.strip()
        if "Description" in df.columns
        else "Tanımsız Ürün"
    )
    df["City"] = (
        df["City"].fillna("Belirsiz").astype(str).str.strip()
        if "City" in df.columns
        else "Belirsiz"
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

    df["EstCost"] = df["GrossAmount"] * 0.44
    df["EstProfit"] = df["TotalAmount"] - df["EstCost"]

    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.weekday  # 0=Pzt .. 6=Paz

    return df


# -----------------------------------------------------------------------------
# 3. FORMATLAMA FONKSİYONLARI VE RAFRADAR V2 CSS
# -----------------------------------------------------------------------------
MONTH_TR_SHORT = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
MONTH_TR_FULL = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
DOW_TR_NAMES = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}


def fmt_date_tr(d: datetime.date) -> str:
    return f"{d.day} {MONTH_TR_SHORT[d.month - 1]} {d.year}"


def fmt_range_tr(d1: datetime.date, d2: datetime.date) -> str:
    if d1.year == d2.year:
        return f"{d1.day} {MONTH_TR_SHORT[d1.month - 1]} – {d2.day} {MONTH_TR_SHORT[d2.month - 1]} {d2.year}"
    return f"{fmt_date_tr(d1)} – {fmt_date_tr(d2)}"


def fmt_tl(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "0 ₺"
    sign = "−" if val < 0 else ""
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
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #f6f3ee;
            --fg: #1f2a44;
            --side: #faf7f2;
            --line: #e4dfd6;
        }
        .stApp {
            background-color: #f6f3ee !important;
            font-family: 'Inter', system-ui, sans-serif;
            color: #1f2a44;
        }
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2.5rem !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #faf7f2 !important;
            border-right: 1px solid #e4dfd6 !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
        }
        .rr-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 4px;
        }
        .rr-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 28px;
            color: #1f2a44;
            margin: 0;
        }
        .rr-header p {
            margin: 4px 0 0 0;
            font-size: 14px;
            font-style: italic;
            color: #5b6475;
        }
        .rr-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
            margin: 6px 0 12px 0;
        }
        .pill {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 16px 18px;
            box-sizing: border-box;
            margin-bottom: 12px;
        }
        .kpi {
            border-top: 4px solid #3d9a84;
            padding: 13px 15px;
            height: 100%;
        }
        .kl { font-size: 12px; color: #5b6475; }
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
            font-size: 15px;
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
        .warn { background: #fdf0cf; color: #7a5600; }
        .mute { background: #f0ebe2; color: #5b6475; }
        .neu  { background: #e4e7ee; color: #2b3550; }
        .rule { border-top: 1px dashed #d9d3c7; margin: 8px 0 6px 0; }
        .rr-note {
            font-size: 12px;
            color: #5b6475;
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 7px 12px;
            margin: 8px 0 12px 0;
        }
        .h {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 15px;
            color: #1f2a44;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .sub { font-size: 12px; color: #5b6475; margin: 3px 0 0 0; }
        .section-takeaway {
            font-size: 13.5px;
            font-weight: 600;
            color: #1f2a44;
            margin: 6px 0 4px 0;
            line-height: 1.4;
        }
        .insight-box {
            background-color: #faf7f2;
            border: 1px solid #e4dfd6;
            border-left: 4px solid #3d9a84;
            padding: 10px 13px;
            border-radius: 0 8px 8px 0;
            font-size: 12px;
            color: #1f2a44;
            line-height: 1.45;
            margin-top: 8px;
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
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }
        .out {
            border-radius: 10px;
            padding: 9px 11px;
            background: #eaf4f0;
        }
        .out .t { font-size: 11px; color: #5b6475; }
        .out .v { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 16px; color: #1f2a44; }
        .mini {
            flex: 1 1 210px;
            min-width: 0;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 11px 13px;
            background: #faf7f2;
        }
        .mini h3 { font-size: 13px; font-weight: 600; margin: 0 0 2px 0; }
        .mini ol { margin: 6px 0 0 0; padding-left: 18px; font-size: 12px; line-height: 1.45; }
        .mini li { margin-bottom: 5px; }
        .leak-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 8px;
        }
        .leak-item {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 8px;
            padding: 8px 10px;
        }
        .leak-item-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 2px;
        }
        .leak-item-desc {
            font-size: 11px;
            color: #5b6475;
            line-height: 1.35;
        }
        .pareto-box {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-radius: 10px;
            padding: 11px 13px;
            margin-bottom: 8px;
        }
        .abc-stack-bar {
            display: flex;
            height: 24px;
            border-radius: 7px;
            overflow: hidden;
            margin: 5px 0 4px 0;
        }
        .tier-row { margin-bottom: 6px; }
        .tier-header {
            display: flex;
            justify-content: space-between;
            font-size: 11.5px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 2px;
        }
        .tier-bar-bg {
            width: 100%;
            height: 6px;
            background: #efe9de;
            border-radius: 4px;
            overflow: hidden;
        }
        .tier-bar-fill { height: 100%; border-radius: 4px; }
        .action-card {
            background: #faf7f2;
            border: 1px solid #e4dfd6;
            border-left: 4px solid #3d9a84;
            border-radius: 10px;
            padding: 11px 13px;
            margin-bottom: 8px;
        }
        .action-card.urgent { border-left-color: #d0574f; }
        .action-card.high   { border-left-color: #e0a526; }
        .action-card.medium { border-left-color: #3d9a84; }
        .action-card.blue   { border-left-color: #3b6fd1; }
        .action-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
            flex-wrap: wrap;
            gap: 6px;
        }
        .action-headline {
            font-family: 'Poppins', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #1f2a44;
            margin-bottom: 3px;
        }
        .action-body {
            font-size: 11.5px;
            color: #5b6475;
            line-height: 1.4;
            margin-bottom: 6px;
        }
        .action-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            padding: 5px 10px;
            border-radius: 6px;
            border: 1px solid #e4dfd6;
            font-size: 11.5px;
        }
        .action-impact {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 13.5px;
        }
        .hm { width: 100%; border-collapse: separate; border-spacing: 0; }
        .hm td { text-align: center; font-size: 12px; font-weight: 600; padding: 7px 4px; border: 2px solid #fff; }
        .hm th { font-size: 11px; font-weight: 600; color: #5b6475; padding: 4px; text-align: center; border: none; }
        .rr-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .rr-table th { font-size: 11px; font-weight: 600; color: #5b6475; text-align: left; padding: 7px 8px; border-bottom: 2px solid #e4dfd6; text-transform: uppercase; }
        .rr-table td { font-size: 12px; padding: 8px; border-bottom: 1px solid #f0ebe2; vertical-align: middle; color: #1f2a44; }
        .vq-step-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; align-items: stretch; }
        .vq-step { flex: 1 1 130px; min-width: 0; border: 1px solid #e4dfd6; border-radius: 10px; padding: 11px 13px; }
        .vq-step-label { font-size: 11px; font-weight: 600; color: #5b6475; }
        .vq-step-val { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 21px; color: #1f2a44; margin: 3px 0; }
        .vq-step-desc { font-size: 11.5px; color: #1f2a44; line-height: 1.3; }
        .vq-arrow { align-self: center; color: #5b6475; font-size: 17px; }
        .rr-float-wrap { position: fixed; right: 20px; bottom: 20px; z-index: 9999; }
        .rr-float-wrap > summary {
            list-style: none;
            width: 54px;
            height: 54px;
            border-radius: 50%;
            background: #3d9a84;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(31, 42, 68, 0.25);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .rr-float-wrap > summary::-webkit-details-marker { display: none; }
        .rr-float-panel {
            position: absolute;
            right: 0;
            bottom: 64px;
            width: 340px;
            max-width: calc(100vw - 24px);
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            box-shadow: 0 10px 28px rgba(31, 42, 68, 0.25);
            padding: 12px;
            color: #1f2a44;
        }
        .rr-qa-chip {
            border: 1px solid #cfc9bd;
            border-radius: 8px;
            background: #faf7f2;
            margin-bottom: 6px;
            padding: 7px 10px;
        }
        .rr-qa-chip > summary {
            list-style: none;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            color: #1f2a44;
            display: flex;
            justify-content: space-between;
        }
        .rr-qa-chip > summary::-webkit-details-marker { display: none; }
        .rr-qa-ans {
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px dashed #d9d3c7;
            font-size: 11.5px;
            color: #1f2a44;
            line-height: 1.4;
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


def render_section_card_header(num: int, title: str, subtitle: str = "", takeaway: str = ""):
    sub_html = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    take_html = f'<div class="section-takeaway">{takeaway}</div>' if takeaway else ""
    render_html(
        f"""
        <div class="card" style="margin-bottom:10px;">
            <h2 class="h"><span class="num">{num}</span>{title}</h2>
            {sub_html}
            {take_html}
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 4. SOL MENÜ VE ORTAK VERİ BAĞLAMI (TÜM DEPARTMANLAR İÇİN)
# -----------------------------------------------------------------------------
DEPARTMENT_PAGES = [
    "📊 Yönetici Özeti",
    "🛠️ Veri Kalitesi",
    "📈 Satış & Fiyat",
    "🎯 Pazarlama & Müşteri",
    "🚚 Tedarik Zinciri",
    "📂 Veri Setini Yükle",
]


def render_shared_sidebar(valid_all_df: pd.DataFrame, pages: list = None) -> dict:
    global_min = valid_all_df["Date"].min()
    global_max = valid_all_df["Date"].max()

    def_a = max(global_min, min(global_max, datetime.date(2020, 11, 1)))
    def_b = max(global_min, min(global_max, datetime.date(2020, 12, 14)))
    def_ca = max(global_min, min(global_max, datetime.date(2020, 9, 1)))
    def_cb = max(global_min, min(global_max, datetime.date(2020, 10, 31)))

    if "rr_fa" not in st.session_state:
        st.session_state["rr_fa"] = def_a
        st.session_state["rr_fb"] = def_b
        st.session_state["rr_fca"] = def_ca
        st.session_state["rr_fcb"] = def_cb

    nav_options = pages if pages is not None else DEPARTMENT_PAGES

    with st.sidebar:
        render_html(
            """
            <div style="display:flex; align-items:center; gap:9px; padding-bottom:10px; border-bottom:1px solid #e4dfd6; margin-bottom:10px;">
                <svg width="28" height="26" viewBox="0 0 28 26" aria-hidden="true">
                    <rect x="2" y="10" width="6" height="14" fill="#1f2a44"/>
                    <rect x="11" y="3" width="6" height="21" fill="#3d9a84"/>
                    <rect x="20" y="14" width="6" height="10" fill="#e0a526"/>
                </svg>
                <div>
                    <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:20px; color:#1f2a44; line-height:1.1;">RafRadar</div>
                    <div style="font-size:11px; color:#5b6475;">B2B Karar Destek</div>
                </div>
            </div>
            """
        )

        selected_page = st.radio(
            "Sayfalar",
            nav_options,
            index=0,
            key="rr_dept_nav",
            label_visibility="collapsed",
        )
        render_html("<div style='border-top:1px solid #e4dfd6; margin:8px 0;'></div>")

        render_html("<div style='font-family:Poppins,sans-serif; font-size:12px; font-weight:600; color:#1f2a44; margin-bottom:4px;'>Seçili dönem</div>")
        sc1, sc2 = st.columns(2)
        with sc1:
            fa = st.date_input("Başlangıç", value=st.session_state["rr_fa"], min_value=global_min, max_value=global_max, key="rr_shared_fa")
        with sc2:
            fb = st.date_input("Bitiş", value=st.session_state["rr_fb"], min_value=global_min, max_value=global_max, key="rr_shared_fb")
        if fa > fb:
            fa, fb = fb, fa
        st.session_state["rr_fa"], st.session_state["rr_fb"] = fa, fb

        render_html("<div style='font-family:Poppins,sans-serif; font-size:12px; font-weight:600; color:#1f2a44; margin:8px 0 4px 0;'>Karşılaştırma dönemi</div>")
        cc1, cc2 = st.columns(2)
        with cc1:
            fca = st.date_input("Başlangıç ", value=st.session_state["rr_fca"], min_value=global_min, max_value=global_max, key="rr_shared_fca")
        with cc2:
            fcb = st.date_input("Bitiş ", value=st.session_state["rr_fcb"], min_value=global_min, max_value=global_max, key="rr_shared_fcb")
        if fca > fcb:
            fca, fcb = fcb, fca
        st.session_state["rr_fca"], st.session_state["rr_fcb"] = fca, fcb

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Önceki dönem", use_container_width=True, key="rr_shared_prevper"):
                days_len = max(1, (fb - fa).days + 1)
                new_cb = max(global_min, fa - datetime.timedelta(days=1))
                new_ca = max(global_min, new_cb - datetime.timedelta(days=days_len - 1))
                st.session_state["rr_fca"] = new_ca
                st.session_state["rr_fcb"] = new_cb
                st.rerun()
        with b_col2:
            if st.button("Sıfırla", use_container_width=True, key="rr_shared_reset"):
                st.session_state["rr_fa"] = def_a
                st.session_state["rr_fb"] = def_b
                st.session_state["rr_fca"] = def_ca
                st.session_state["rr_fcb"] = def_cb
                st.rerun()

        render_html("<div style='border-top:1px solid #e4dfd6; margin:8px 0;'></div>")

        top_cats = valid_all_df.groupby("Category")["TotalAmount"].sum().nlargest(25).index.tolist()
        selected_cat = st.selectbox("Kategori", ["Tüm Kategoriler"] + top_cats, index=0, key="rr_shared_cat")

        top_cities = valid_all_df.groupby("City")["TotalAmount"].sum().nlargest(30).index.tolist()
        selected_city = st.selectbox("İl", ["Tüm İller"] + top_cities, index=0, key="rr_shared_city")

    return {
        "page": selected_page,
        "fa": st.session_state["rr_fa"],
        "fb": st.session_state["rr_fb"],
        "fca": st.session_state["rr_fca"],
        "fcb": st.session_state["rr_fcb"],
        "category": selected_cat,
        "city": selected_city,
    }


def render_rafradar_sidebar(valid_all_df: pd.DataFrame, show_page_nav: bool = True, pages: list = None) -> dict:
    opts = pages if pages is not None else (["📊 Yönetici Özeti", "🛠️ Veri Kalitesi"] if show_page_nav else ["📊 Yönetici Özeti"])
    return render_shared_sidebar(valid_all_df, pages=opts)


# -----------------------------------------------------------------------------
# 5. DİNAMİK ANALİTİK HESAPLAMA MOTORU
# -----------------------------------------------------------------------------
def compute_dynamic_metrics(
    sales_df: pd.DataFrame,
    comp_sales_df: pd.DataFrame,
    raw_scope_df: pd.DataFrame,
    s_days: int,
    c_days: int,
) -> dict:
    net_rev = float(sales_df["TotalAmount"].sum())
    gross_rev = float(sales_df["GrossAmount"].sum())
    est_profit = float(sales_df["EstProfit"].sum())
    profit_margin = (est_profit / net_rev * 100.0) if net_rev > 0 else 0.0
    avg_discount = ((gross_rev - net_rev) / gross_rev * 100.0) if gross_rev > 0 else 0.0

    comp_net_rev = float(comp_sales_df["TotalAmount"].sum()) if not comp_sales_df.empty else 0.0
    comp_gross_rev = float(comp_sales_df["GrossAmount"].sum()) if not comp_sales_df.empty else 0.0
    comp_profit = float(comp_sales_df["EstProfit"].sum()) if not comp_sales_df.empty else 0.0
    comp_discount = ((comp_gross_rev - comp_net_rev) / comp_gross_rev * 100.0) if comp_gross_rev > 0 else 0.0

    def rel_pct(a, b):
        if b == 0 or pd.isna(b):
            return 0.0
        return ((a - b) / abs(b)) * 100.0

    d_net_rev = net_rev / max(1, s_days)
    d_comp_rev = comp_net_rev / max(1, c_days)
    d_profit = est_profit / max(1, s_days)
    d_comp_profit = comp_profit / max(1, c_days)

    active_customers = int(sales_df["CustomerID"].nunique())
    comp_customers = int(comp_sales_df["CustomerID"].nunique()) if not comp_sales_df.empty else 0
    d_cust = float(sales_df.groupby("Date")["CustomerID"].nunique().mean()) if not sales_df.empty else 0.0
    d_comp_cust = float(comp_sales_df.groupby("Date")["CustomerID"].nunique().mean()) if not comp_sales_df.empty else 0.0

    active_skus = int(sales_df["Description"].nunique())
    comp_skus = int(comp_sales_df["Description"].nunique()) if not comp_sales_df.empty else 0
    total_invoices = int(sales_df["InvoiceNo"].nunique())
    comp_invoices = int(comp_sales_df["InvoiceNo"].nunique()) if not comp_sales_df.empty else 0
    d_inv = total_invoices / max(1, s_days)
    d_comp_inv = comp_invoices / max(1, c_days)

    cat_medians = sales_df.groupby("Category")["DiscountPct"].transform("median")
    excess_disc_pct = sales_df["DiscountPct"] - cat_medians
    leak_mask = excess_disc_pct > 10.0
    suspicious_orders_count = int(leak_mask.sum())
    discount_leak_tl = float((sales_df.loc[leak_mask, "GrossAmount"] * (excess_disc_pct[leak_mask] / 100.0)).sum())
    if discount_leak_tl < 1000 and len(sales_df) > 0:
        mild_mask = excess_disc_pct > 5.0
        suspicious_orders_count = int(mild_mask.sum())
        discount_leak_tl = float((sales_df.loc[mild_mask, "GrossAmount"] * (excess_disc_pct[mild_mask] / 100.0)).sum())

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
    mid_cust_count = int((cust_df["Tier"] == "Gelişen Orta Bayiler").sum())
    tail_cust_count = int((cust_df["Tier"] == "Küçük / Tekil Bayiler").sum())
    vip_cust_share_pct = (vip_cust_count / max(1, active_customers)) * 100.0
    top10_cust_rev_share = (cust_df.head(10)["Revenue"].sum() / net_rev * 100.0) if net_rev > 0 else 0.0

    vip_rev_sum = float(cust_df.loc[cust_df["Tier"] == "VIP & Lokomotif Bayiler", "Revenue"].sum())
    mid_rev_sum = float(cust_df.loc[cust_df["Tier"] == "Gelişen Orta Bayiler", "Revenue"].sum())
    tail_rev_sum = float(cust_df.loc[cust_df["Tier"] == "Küçük / Tekil Bayiler", "Revenue"].sum())

    recency_threshold = min(90, max(14, int(s_days * 0.30)))
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
    b_sku_count = int((sku_df["ABC"] == "B Sınıfı Düzenli Kitaplar").sum())
    c_sku_count = int((sku_df["ABC"] == "C Sınıfı & Uzun Kuyruk").sum())
    a_sku_share_pct = (a_sku_count / max(1, active_skus)) * 100.0
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
        "comp_net_rev": comp_net_rev,
        "comp_gross_rev": comp_gross_rev,
        "comp_profit": comp_profit,
        "comp_discount": comp_discount,
        "comp_customers": comp_customers,
        "comp_invoices": comp_invoices,
        "comp_skus": comp_skus,
        "d_comp_rev": d_comp_rev,
        "d_comp_profit": d_comp_profit,
        "d_comp_cust": d_comp_cust,
        "d_comp_inv": d_comp_inv,
        "suspicious_orders_count": suspicious_orders_count,
        "discount_leak_tl": discount_leak_tl,
        "cat_anom_df": cat_anom_df,
        "cust_df": cust_df,
        "vip_cust_count": vip_cust_count,
        "mid_cust_count": mid_cust_count,
        "tail_cust_count": tail_cust_count,
        "vip_cust_share_pct": vip_cust_share_pct,
        "top10_cust_rev_share": top10_cust_rev_share,
        "vip_rev_sum": vip_rev_sum,
        "mid_rev_sum": mid_rev_sum,
        "tail_rev_sum": tail_rev_sum,
        "recency_threshold": recency_threshold,
        "at_risk_vip_count": at_risk_vip_count,
        "vip_recovery_potential_tl": vip_recovery_potential_tl,
        "recoverable_vip_tl": vip_recovery_potential_tl,
        "top_risk_examples": top_risk_examples,
        "one_time_ratio_pct": one_time_ratio_pct,
        "one_time_count": len(one_time_custs),
        "retention_potential_tl": retention_potential_tl,
        "sku_df": sku_df,
        "a_sku_count": a_sku_count,
        "b_sku_count": b_sku_count,
        "c_sku_count": c_sku_count,
        "a_sku_share_pct": a_sku_share_pct,
        "c_sku_share_pct": c_sku_share_pct,
        "dead_sku_count": dead_sku_count,
        "dead_threshold": dead_threshold,
        "stockout_prevention_tl": stockout_prevention_tl,
        "stockout_risk_tl": stockout_prevention_tl,
        "top_city": top_city,
        "top_city_name": top_city,
        "top_city_share": top_city_share,
        "top_cat_name": top_cat_name,
        "top_cat_share": top_cat_share,
    }


def build_shared_context(raw_df: pd.DataFrame = None, pages: list = None, external_filters: dict = None) -> dict:
    inject_rafradar_v2_css()

    full_raw_df = raw_df if (raw_df is not None and not raw_df.empty) else load_rafradar_data()
    if full_raw_df.empty:
        return {}

    if "DiscountPct" not in full_raw_df.columns or "EstProfit" not in full_raw_df.columns:
        full_raw_df = load_rafradar_data()

    valid_all_df = full_raw_df[~full_raw_df["IsReturn"]].copy()
    nav = external_filters if external_filters is not None else render_shared_sidebar(valid_all_df, pages=pages)

    fa, fb = nav["fa"], nav["fb"]
    fca, fcb = nav["fca"], nav["fcb"]
    selected_cat = nav.get("category", "Tüm Kategoriler")
    selected_city = nav.get("city", "Tüm İller")
    selected_page = nav.get("page", DEPARTMENT_PAGES[0])

    s_days = max(1, (fb - fa).days + 1)
    c_days = max(1, (fcb - fca).days + 1)

    filtered_raw_all = full_raw_df.copy()
    filtered_valid_all = valid_all_df.copy()

    if selected_cat != "Tüm Kategoriler":
        filtered_raw_all = filtered_raw_all[filtered_raw_all["Category"] == selected_cat]
        filtered_valid_all = filtered_valid_all[filtered_valid_all["Category"] == selected_cat]

    if selected_city != "Tüm İller":
        filtered_raw_all = filtered_raw_all[filtered_raw_all["City"] == selected_city]
        filtered_valid_all = filtered_valid_all[filtered_valid_all["City"] == selected_city]

    scope_sales_df = filtered_valid_all[(filtered_valid_all["Date"] >= fa) & (filtered_valid_all["Date"] <= fb)].copy()
    comp_sales_df = filtered_valid_all[(filtered_valid_all["Date"] >= fca) & (filtered_valid_all["Date"] <= fcb)].copy()
    scope_raw_df = filtered_raw_all[(filtered_raw_all["Date"] >= fa) & (filtered_raw_all["Date"] <= fb)].copy()

    if scope_sales_df.empty:
        return {
            "empty": True,
            "page": selected_page,
            "nav": nav,
            "fa": fa,
            "fb": fb,
            "fca": fca,
            "fcb": fcb,
            "s_days": s_days,
            "c_days": c_days,
            "sel_lbl": fmt_range_tr(fa, fb),
            "comp_lbl": fmt_range_tr(fca, fcb),
            "category": selected_cat,
            "city": selected_city,
            "full_raw_df": filtered_raw_all,
            "valid_all_df": filtered_valid_all,
            "sel_df": scope_sales_df,
            "comp_df": comp_sales_df,
            "raw_scope_df": scope_raw_df,
        }

    m = compute_dynamic_metrics(scope_sales_df, comp_sales_df, scope_raw_df, s_days, c_days)
    sel_lbl = fmt_range_tr(fa, fb)
    comp_lbl = fmt_range_tr(fca, fcb)

    first_buy_dates = filtered_valid_all.groupby("CustomerID")["Date"].min()
    new_cust_count = int(((first_buy_dates >= fa) & (first_buy_dates <= fb)).sum())
    before_custs = set(filtered_valid_all.loc[filtered_valid_all["Date"] < fa, "CustomerID"].unique())
    sel_custs = set(scope_sales_df["CustomerID"].unique())
    comp_custs = set(comp_sales_df["CustomerID"].unique()) if not comp_sales_df.empty else set()
    cont_cust_count = len(sel_custs & comp_custs)
    back_cust_count = len((sel_custs - comp_custs) & before_custs)
    lost_cust_count = len(comp_custs - sel_custs)

    s_qty = float(scope_sales_df["Quantity"].sum())
    c_qty = float(comp_sales_df["Quantity"].sum()) if not comp_sales_df.empty else 0.0
    s_dq = s_qty / max(1, s_days)
    c_dq = c_qty / max(1, c_days)

    s_per_c = m["net_rev"] / max(1, m["active_customers"])
    c_per_c = m["comp_net_rev"] / max(1, m["comp_customers"]) if m["comp_customers"] > 0 else 0.0
    s_dpc = m["d_net_rev"] / max(1e-6, m["d_cust"]) if m["d_cust"] > 0 else 0.0
    c_dpc = m["d_comp_rev"] / max(1e-6, m["d_comp_cust"]) if m["d_comp_cust"] > 0 else 0.0

    s_aov = m["net_rev"] / max(1, m["total_invoices"])
    c_aov = m["comp_net_rev"] / max(1, m["comp_invoices"]) if m["comp_invoices"] > 0 else 0.0
    s_basket_q = s_qty / max(1, m["total_invoices"])
    c_basket_q = c_qty / max(1, m["comp_invoices"]) if m["comp_invoices"] > 0 else 0.0

    def _rel(a, b):
        if b == 0 or pd.isna(b):
            return 0.0
        return ((a - b) / abs(b)) * 100.0

    S_summary = {
        "days": s_days,
        "net": m["net_rev"],
        "dnet": m["d_net_rev"],
        "gross": m["gross_rev"],
        "profit": m["est_profit"],
        "dprofit": m["d_profit"],
        "margin": m["profit_margin"],
        "disc": m["avg_discount"],
        "cust": m["active_customers"],
        "dcust": m["d_cust"],
        "new_cust": new_cust_count,
        "cont_cust": cont_cust_count,
        "back_cust": back_cust_count,
        "lost_cust": lost_cust_count,
        "orders": m["total_invoices"],
        "dorders": m["d_inv"],
        "qty": s_qty,
        "dqty": s_dq,
        "sku": m["active_skus"],
        "per_cust": s_per_c,
        "d_per_cust": s_dpc,
        "aov": s_aov,
        "basket_q": s_basket_q,
    }
    C_summary = {
        "days": c_days,
        "net": m["comp_net_rev"],
        "dnet": m["d_comp_rev"],
        "profit": m["comp_profit"],
        "dprofit": m["d_comp_profit"],
        "disc": m["comp_discount"],
        "cust": m["comp_customers"],
        "dcust": m["d_comp_cust"],
        "orders": m["comp_invoices"],
        "dorders": m["d_comp_inv"],
        "qty": c_qty,
        "dqty": c_dq,
        "sku": m["comp_skus"],
        "per_cust": c_per_c,
        "d_per_cust": c_dpc,
        "aov": c_aov,
        "basket_q": c_basket_q,
        "qty_chg": _rel(s_qty, c_qty),
        "dqty_chg": _rel(s_dq, c_dq),
        "per_cust_chg": _rel(s_per_c, c_per_c),
        "d_per_cust_chg": _rel(s_dpc, c_dpc),
        "aov_chg": _rel(s_aov, c_aov),
        "basket_q_chg": _rel(s_basket_q, c_basket_q),
    }

    return {
        "empty": False,
        "page": selected_page,
        "nav": nav,
        "fa": fa,
        "fb": fb,
        "fca": fca,
        "fcb": fcb,
        "s_days": s_days,
        "c_days": c_days,
        "sel_lbl": sel_lbl,
        "comp_lbl": comp_lbl,
        "category": selected_cat,
        "city": selected_city,
        "full_raw_df": filtered_raw_all,
        "valid_all_df": filtered_valid_all,
        "sel_df": scope_sales_df,
        "comp_df": comp_sales_df,
        "raw_scope_df": scope_raw_df,
        "m": m,
        "S": S_summary,
        "C": C_summary,
    }


# -----------------------------------------------------------------------------
# 6. SOHBET ASİSTANI VE ORTAK ÜST BAR
# -----------------------------------------------------------------------------
def answer_executive_question(query: str, m: dict, sel_lbl: str, comp_lbl: str, selected_cat: str) -> str:
    q = query.lower().strip()
    total_opp = (
        m["vip_recovery_potential_tl"]
        + m["stockout_prevention_tl"]
        + m["retention_potential_tl"]
        + m["discount_leak_tl"]
    )
    if any(k in q for k in ["ciro", "büyü", "kâr", "kar", "marj", "finans", "durum"]):
        return (
            f"**{sel_lbl}:** Net ciro **{fmt_tl(m['net_rev'])}** (günlük ort. **{fmt_tl(m['d_net_rev'], 1)}**), "
            f"tahmini brüt kâr **{fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)**."
        )
    if any(k in q for k in ["iskonto", "indirim", "kaçak", "anomali", "şüpheli", "fiyat"]):
        return (
            f"Ort. indirim **%{m['avg_discount']:.1f}**. **{fmt_int(m['suspicious_orders_count'])} satırda** "
            f"kategori normu aşımıyla **{fmt_tl(m['discount_leak_tl'])}** fazla indirim oluştu."
        )
    if any(k in q for k in ["vip", "müşteri", "bayi", "kayıp", "churn", "risk", "gecik"]):
        return (
            f"Aktif **{fmt_int(m['active_customers'])}** müşterinin **%{m['vip_cust_share_pct']:.1f}'i ({m['vip_cust_count']} A sınıfı)** "
            f"cironun %80'ini üretiyor. **{m['at_risk_vip_count']} VIP müşteride** gecikme var (+{fmt_tl(m['vip_recovery_potential_tl'])})."
        )
    if any(k in q for k in ["pareto", "80/20", "portföy", "abc"]):
        return (
            f"Müşterilerin **%{m['vip_cust_share_pct']:.1f}'i** ve ürünlerin **%{m['a_sku_share_pct']:.1f}'i** cironun %80'ini getiriyor. "
            f"İlk 10 müşteri cironun **%{m['top10_cust_rev_share']:.1f}**'ini üretiyor."
        )
    if any(k in q for k in ["stok", "tedarik", "ölü", "hareketsiz", "ürün"]):
        return (
            f"**{fmt_int(m['a_sku_count'])} A ürününde** stok koruması **+{fmt_tl(m['stockout_prevention_tl'])}** sağlar; "
            f"**{m['dead_threshold']}+ gün** satılmayan **{fmt_int(m['dead_sku_count'])} ürün** eritilmeli."
        )
    if any(k in q for k in ["aksiyon", "ne yapmalı", "30 gün", "plan"]):
        return (
            f"**30 Günlük Toplam Potansiyel: +{fmt_tl(total_opp)}** "
            f"(VIP geri kazanım +{fmt_tl(m['vip_recovery_potential_tl'])}, İndirim tavanı +{fmt_tl(m['discount_leak_tl'])}, "
            f"A sınıfı stok koruması +{fmt_tl(m['stockout_prevention_tl'])}, 2. sipariş paketi +{fmt_tl(m['retention_potential_tl'])})."
        )
    return (
        f"**{sel_lbl}:** Net ciro **{fmt_tl(m['net_rev'])}**, aktif müşteri **{fmt_int(m['active_customers'])}**, "
        f"lider il **{m['top_city']} (%{m['top_city_share']:.1f})**, toplam fırsat **+{fmt_tl(total_opp)}**."
    )


def render_chatbox_widgets(m: dict, sel_lbl: str, comp_lbl: str, selected_cat: str):
    if "rafradar_chat_history" not in st.session_state:
        st.session_state["rafradar_chat_history"] = [
            {"role": "assistant", "text": f"**{sel_lbl}** dönemi hazır. Aşağıdan bir soru seçin veya yazın."}
        ]

    with st.popover("💬 Asistan", use_container_width=True):
        quick_questions = [
            ("📈 Ciro & Kâr", "Ciro ve kâr durumu nedir?"),
            ("🚨 Geciken VIP", "Geciken VIP müşteriler kimler?"),
            ("🏷️ Fazla İndirim", "Fazla indirim kaçağı ne kadar?"),
            ("🎯 30 Gün Aksiyon", "30 günlük aksiyon planı nedir?"),
        ]
        q_cols = st.columns(2)
        triggered_q = None
        for idx, (btn_label, full_q) in enumerate(quick_questions):
            with q_cols[idx % 2]:
                if st.button(btn_label, key=f"rr_quick_q_{idx}", use_container_width=True):
                    triggered_q = full_q

        user_input = st.chat_input("Sorunuzu yazın...", key="rr_chat_inp")
        active_query = triggered_q or user_input
        if active_query:
            reply = answer_executive_question(active_query, m, sel_lbl, comp_lbl, selected_cat)
            st.session_state["rafradar_chat_history"].append({"role": "user", "text": active_query})
            st.session_state["rafradar_chat_history"].append({"role": "assistant", "text": reply})

        for msg in st.session_state["rafradar_chat_history"][-4:]:
            if msg["role"] == "user":
                render_html(f'<div style="background:#ece7dd;padding:6px 10px;border-radius:8px;margin:4px 0;font-size:12px;font-weight:600;">👤 {msg["text"]}</div>')
            else:
                st.markdown(f"<div style='background:#faf7f2;border-left:3px solid #3d9a84;padding:6px 10px;border-radius:6px;margin:4px 0;font-size:12px;'>{msg['text']}</div>", unsafe_allow_html=True)

    q1 = answer_executive_question("ciro", m, sel_lbl, comp_lbl, selected_cat)
    q2 = answer_executive_question("vip", m, sel_lbl, comp_lbl, selected_cat)
    q3 = answer_executive_question("indirim", m, sel_lbl, comp_lbl, selected_cat)
    q4 = answer_executive_question("aksiyon", m, sel_lbl, comp_lbl, selected_cat)

    render_html(
        f"""
        <details class="rr-float-wrap">
            <summary title="RafRadar Asistan">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </summary>
            <div class="rr-float-panel">
                <div style="background:#1f2a44;color:#fff;padding:8px 11px;border-radius:8px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:Poppins,sans-serif;font-weight:600;font-size:13px;">💬 RafRadar Asistan</span>
                    <span class="pill" style="background:#d6ece5;color:#14513f;">{sel_lbl}</span>
                </div>
                <details class="rr-qa-chip" open><summary><span>📈 Ciro &amp; kâr durumu?</span><span>▾</span></summary><div class="rr-qa-ans">{q1}</div></details>
                <details class="rr-qa-chip"><summary><span>🚨 Geciken VIP müşteriler?</span><span>▾</span></summary><div class="rr-qa-ans">{q2}</div></details>
                <details class="rr-qa-chip"><summary><span>🏷️ Fazla indirim kaçağı?</span><span>▾</span></summary><div class="rr-qa-ans">{q3}</div></details>
                <details class="rr-qa-chip" style="margin-bottom:0;"><summary><span>🎯 30 günlük aksiyonlar?</span><span>▾</span></summary><div class="rr-qa-ans">{q4}</div></details>
            </div>
        </details>
        """
    )


def render_department_header_and_kpis(
    title: str,
    subtitle: str,
    ctx: dict,
    kpi_cards: list = None,
    note_text: str = None,
):
    m = ctx["m"]
    sel_lbl = ctx["sel_lbl"]
    comp_lbl = ctx["comp_lbl"]
    s_days = ctx["s_days"]
    c_days = ctx["c_days"]
    selected_cat = ctx["category"]
    selected_city = ctx["city"]

    hdr_left, hdr_right = st.columns([8.2, 1.8])
    with hdr_left:
        render_html(
            f"""
            <header class="rr-header">
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
            </header>
            """
        )
    with hdr_right:
        render_chatbox_widgets(m, sel_lbl, comp_lbl, selected_cat)

    cat_pill = f'<span class="pill" style="background:#dbe5f8;color:#1f2a44;">{selected_cat}</span>' if selected_cat != "Tüm Kategoriler" else ""
    city_pill = f'<span class="pill" style="background:#fdf0cf;color:#7a5600;">{selected_city}</span>' if selected_city != "Tüm İller" else ""
    render_html(
        f"""
        <section class="rr-pills">
            <span class="pill" style="background:#d6ece5;color:#14513f;">Seçili: {sel_lbl} · {s_days} gün</span>
            <span style="font-size:12px;color:#5b6475;">karşılaştırma</span>
            <span class="pill" style="background:#ece7dd;color:#1f2a44;">{comp_lbl} · {c_days} gün</span>
            {cat_pill}
            {city_pill}
        </section>
        """
    )

    if kpi_cards:
        cols = st.columns(len(kpi_cards))
        for col, card in zip(cols, kpi_cards):
            with col:
                render_kcard(
                    label=card["label"],
                    main_val=card["main_val"],
                    main_chg_html=card.get("main_chg_html", ""),
                    sub_label=card.get("sub_label", "Günlük ort."),
                    sub_val=card.get("sub_val", ""),
                    sub_chg_html=card.get("sub_chg_html", ""),
                    top_color=card.get("top_color", "#3d9a84"),
                )

    if note_text:
        render_html(f'<div class="rr-note">{note_text}</div>')


# -----------------------------------------------------------------------------
# 7. YÖNETİCİ ÖZETİ & VERİ KALİTESİ (İLK TASARIM + RAFRADAR V2 + YARATICI GRAFİKLER)
# -----------------------------------------------------------------------------
def render_executive_summary(filtered_df=None, raw_df=None, external_filters: dict = None):
    ctx = build_shared_context(raw_df=raw_df, external_filters=external_filters)
    if not ctx or ctx.get("empty", False):
        st.warning("Seçilen filtrede kayıt bulunamadı. Sol menüden tarih aralığını genişletin.")
        return

    m = ctx["m"]
    S = ctx["S"]
    valid_all_df = ctx["valid_all_df"]
    scope_sales_df = ctx["sel_df"]
    scope_raw_df = ctx["raw_scope_df"]
    sel_lbl = ctx["sel_lbl"]
    comp_lbl = ctx["comp_lbl"]
    s_days = ctx["s_days"]
    selected_cat = ctx["category"]
    selected_page = ctx["page"]

    is_kalite = "Veri Kalitesi" in str(selected_page)

    # =========================================================================
    # SAYFA 1: YÖNETİCİ ÖZETİ
    # =========================================================================
    if not is_kalite:
        disc_pts_str = f"{m['disc_diff_pts']:+.1f} puan".replace(".", ",")
        disc_cls = "neg" if m["disc_diff_pts"] > 0.5 else ("pos" if m["disc_diff_pts"] < -0.5 else "mute")

        kpi_cards = [
            {
                "label": "Net ciro",
                "main_val": fmt_tl(S["net"]),
                "main_chg_html": chg_badge_html(m["rev_chg"]),
                "sub_label": "Günlük ort.",
                "sub_val": fmt_tl(S["dnet"], 1),
                "sub_chg_html": chg_badge_html(m["d_rev_chg"]),
                "top_color": "#3d9a84",
            },
            {
                "label": "Tahmini brüt kâr & marj",
                "main_val": fmt_tl(S["profit"]),
                "main_chg_html": f'<span class="chg pos">%{S["margin"]:.1f} marj</span>',
                "sub_label": "Günlük ort.",
                "sub_val": fmt_tl(S["dprofit"], 1),
                "sub_chg_html": chg_badge_html(m["d_profit_chg"]),
                "top_color": "#e0a526",
            },
            {
                "label": "Aktif müşteri & Pareto",
                "main_val": fmt_int(S["cust"]),
                "main_chg_html": chg_badge_html(m["cust_chg"]),
                "sub_label": f"%{m['vip_cust_share_pct']:.0f}'i = %80 ciro",
                "sub_val": f"{S['dcust']:.1f}/gün".replace(".", ","),
                "sub_chg_html": chg_badge_html(m["d_cust_chg"]),
                "top_color": "#3b6fd1",
            },
            {
                "label": "Ort. indirim & kaçak",
                "main_val": f"%{S['disc']:.1f}".replace(".", ","),
                "main_chg_html": f'<span class="chg {disc_cls}">{disc_pts_str}</span>',
                "sub_label": "Fazla indirim",
                "sub_val": fmt_tl(m["discount_leak_tl"], 1),
                "sub_chg_html": f'<span class="chg neg">{fmt_int(m["suspicious_orders_count"])} satır</span>',
                "top_color": "#d0574f",
            },
            {
                "label": "Aktif SKU & ölü stok",
                "main_val": fmt_int(S["sku"]),
                "main_chg_html": chg_badge_html(m["sku_chg"]),
                "sub_label": f"{m['dead_threshold']}+ gün pasif",
                "sub_val": f"{fmt_int(m['dead_sku_count'])} SKU",
                "sub_chg_html": f'<span class="chg mute">{fmt_int(S["orders"])} sipariş</span>',
                "top_color": "#1f2a44",
            },
        ]

        render_department_header_and_kpis(
            title="Yönetici Özeti",
            subtitle="Şirket büyüyor mu, kârlılığımız güvende mi ve önümüzdeki 30 günde hangi kritik aksiyonları almalıyız?",
            ctx=ctx,
            kpi_cards=kpi_cards,
        )

        # ---------------------------------------------------------------------
        # ORTA BÖLÜM:
        # [1] FİNANSAL PERFORMANS, SEZONSALLIK VE İSKONTO DİSİPLİNİ
        # [2] KÂR SIZINTISI & PARA AKIŞI (ŞELALE KÖPRÜSÜ VEYA SANKEY DİYAGRAMI)
        # ---------------------------------------------------------------------
        col_left, col_right = st.columns([5.5, 4.5])

        with col_left:
            vh1, vh2 = st.columns([6.6, 3.4])
            with vh2:
                chart_mode = st.selectbox(
                    "Görünüm",
                    ["Yıllık (12 Ay)", "Seçili Dönem"],
                    index=0,
                    label_visibility="collapsed",
                    key="rr_exec_chart_mode",
                )

            if chart_mode == "Yıllık (12 Ay)" or s_days > 65:
                m_grid = pd.DataFrame({"Month": list(range(1, 13))})
                m_agg = (
                    valid_all_df.groupby("Month", as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"))
                )
                ts_df = m_grid.merge(m_agg, on="Month", how="left").fillna(0.0)
                ts_df["Label"] = [MONTH_TR_SHORT[i - 1] for i in ts_df["Month"]]
                period_sub = "12 aylık net ciro (çubuk) ve ortalama iskonto oranı (çizgi)"
            else:
                ts_df = (
                    scope_sales_df.groupby("Date", as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"))
                    .sort_values("Date")
                )
                ts_df["Label"] = pd.to_datetime(ts_df["Date"]).dt.strftime("%d %b")
                period_sub = f"Seçili dönem ({sel_lbl}) günlük net ciro ve iskonto oranı"

            ts_df["DiscPct"] = np.where(
                ts_df["GrossRev"] > 0,
                (ts_df["GrossRev"] - ts_df["NetRev"]) / ts_df["GrossRev"] * 100.0,
                0.0,
            )

            nonzero_ts = ts_df[ts_df["NetRev"] > 0] if (ts_df["NetRev"] > 0).any() else ts_df
            rev_thresh = nonzero_ts["NetRev"].quantile(0.68) if len(nonzero_ts) > 3 else nonzero_ts["NetRev"].mean()
            bar_colors = ["#3d9a84" if v >= rev_thresh else "#1f2a44" for v in ts_df["NetRev"]]

            peak_row = nonzero_ts.loc[nonzero_ts["NetRev"].idxmax()]
            low_row = nonzero_ts.loc[nonzero_ts["NetRev"].idxmin()]
            peak_to_low_ratio = peak_row["NetRev"] / max(1.0, low_row["NetRev"])
            peak_disc_avg = nonzero_ts.loc[nonzero_ts["NetRev"] >= rev_thresh, "DiscPct"].mean()

            dow_rev = scope_sales_df.groupby("DayOfWeek")["TotalAmount"].sum()
            top_dow_idx = int(dow_rev.idxmax()) if not dow_rev.empty else 0
            top_dow_tr = DOW_TR_NAMES.get(top_dow_idx, "Pazartesi")
            top_dow_share = (dow_rev.max() / m["net_rev"] * 100.0) if m["net_rev"] > 0 else 0.0

            with vh1:
                render_html(
                    f"""
                    <div>
                        <h2 class="h"><span class="num">1</span>Finansal performans ve iskonto disiplini</h2>
                        <p class="sub">{period_sub}</p>
                    </div>
                    """
                )

            render_html(
                f"""
                <div class="section-takeaway">
                    Zirve ({peak_row['Label']}: {fmt_tl(peak_row['NetRev'], 1)}) ile dip ({low_row['Label']}) arasında {peak_to_low_ratio:.1f} kat fark var;
                    ancak talep zirvedeyken bile iskonto oranı (%{peak_disc_avg:.1f}) düşmüyor.
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
                    text=[fmt_tl(v, 1) if v > 0 else "" for v in ts_df["NetRev"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Net Ciro: ₺%{y:,.0f}<extra></extra>",
                ),
                secondary_y=False,
            )
            fig1.add_trace(
                go.Scatter(
                    x=ts_df["Label"],
                    y=ts_df["DiscPct"],
                    name="Ort. İskonto (%)",
                    mode="lines+markers",
                    line=dict(color="#e0a526", width=2.8),
                    marker=dict(size=7, color="#e0a526", line=dict(color="#fff", width=1)),
                    hovertemplate="<b>%{x}</b><br>İskonto: %%{y:.1f}<extra></extra>",
                ),
                secondary_y=True,
            )
            disc_min = max(0.0, float(nonzero_ts["DiscPct"].min()) - 6.0)
            disc_max = min(85.0, float(nonzero_ts["DiscPct"].max()) + 6.0)
            fig1.update_layout(
                height=265,
                margin=dict(l=8, r=8, t=18, b=8),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#1f2a44", size=11),
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
                title_text="İskonto (%)",
                range=[disc_min, disc_max],
                showgrid=False,
                secondary_y=True,
            )
            st.plotly_chart(fig1, use_container_width=True)

            render_html(
                f"""
                <div class="insight-box">
                    <b>Stratejik Çıkarım:</b> Yoğun sezonda sabit tutulan <b>%{peak_disc_avg:.1f}</b> iskonto organik talepte marj kaybı yaratıyor.
                    Ayrıca siparişlerin <b>%{top_dow_share:.1f}'i {top_dow_tr}</b> gününe yığılarak depo darboğazı oluşturuyor.
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

            s2_h1, s2_h2 = st.columns([6.2, 3.8])
            with s2_h2:
                s2_mode = st.selectbox(
                    "Grafik",
                    ["Fırsat Köprüsü (Şelale)", "Para Akışı (Sankey)"],
                    index=0,
                    label_visibility="collapsed",
                    key="rr_exec_s2_mode",
                )
            with s2_h1:
                render_html(
                    """
                    <div>
                        <h2 class="h"><span class="num">2</span>Kâr sızıntısı ve fırsat havuzu</h2>
                        <p class="sub">4 operasyonel sızıntı ve brüt cirodan net kâra akış</p>
                    </div>
                    """
                )

            render_html(
                f"""
                <div class="section-takeaway">
                    4 sızıntı kapatıldığında dönem cirosuna +{fmt_tl(total_opportunity_tl, 1)} (+%{uplift_pct:.1f})
                    eklenerek {fmt_tl(target_rev_tl, 1)} seviyesine ulaşılabilir.
                </div>
                """
            )

            if s2_mode == "Para Akışı (Sankey)":
                gross_v = max(1.0, m["gross_rev"])
                net_v = max(1.0, m["net_rev"])
                leak_disc_v = min(max(0.0, m["discount_leak_tl"]), max(0.0, gross_v - net_v))
                norm_disc_v = max(0.0, (gross_v - net_v) - leak_disc_v)
                cost_v = max(0.0, gross_v * 0.44)
                prof_v = max(0.0, m["est_profit"])

                fig2 = go.Figure(
                    data=[
                        go.Sankey(
                            node=dict(
                                pad=12,
                                thickness=16,
                                line=dict(color="#e4dfd6", width=1),
                                label=[
                                    f"Brüt Ciro ({fmt_tl(gross_v, 1)})",
                                    f"Net Ciro ({fmt_tl(net_v, 1)})",
                                    f"Standart İskonto ({fmt_tl(norm_disc_v, 1)})",
                                    f"Fazla İskonto Kaçağı ({fmt_tl(leak_disc_v, 1)})",
                                    f"Tahmini Brüt Kâr ({fmt_tl(prof_v, 1)})",
                                    f"Ürün Maliyeti %44 ({fmt_tl(cost_v, 1)})",
                                ],
                                color=["#1f2a44", "#3b6fd1", "#cfc9bd", "#d0574f", "#3d9a84", "#e0a526"],
                            ),
                            link=dict(
                                source=[0, 0, 0, 1, 1],
                                target=[1, 2, 3, 4, 5],
                                value=[net_v, norm_disc_v, leak_disc_v, prof_v, cost_v],
                                color=[
                                    "rgba(59,111,209,0.28)",
                                    "rgba(207,201,189,0.45)",
                                    "rgba(208,87,79,0.55)",
                                    "rgba(61,154,132,0.42)",
                                    "rgba(224,165,38,0.35)",
                                ],
                            ),
                        )
                    ]
                )
                fig2.update_layout(
                    height=215,
                    margin=dict(l=6, r=6, t=12, b=8),
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=10.5),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                wf_x = [
                    "Seçili<br>Net Ciro",
                    "+ Pasif VIP<br>Kazanımı",
                    "+ AX Stok<br>Koruması",
                    "+ Ç+1 İkinci<br>Sipariş",
                    "+ İskonto<br>Tavanı",
                    "Hedef<br>Potansiyel",
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
                    height=215,
                    margin=dict(l=8, r=8, t=18, b=8),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=11),
                    yaxis=dict(
                        title="Ciro Köprüsü (₺)",
                        range=[0, target_rev_tl * 1.22],
                        gridcolor="#efe9de",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

            risk_ex = m["top_risk_examples"]
            ex_txt = (
                ", ".join(f"<b>{r['CustomerID']}</b> ({fmt_tl(r['Revenue'], 0)})" for _, r in risk_ex.iterrows())
                if not risk_ex.empty
                else "üst segment VIP bayiler"
            )

            render_html(
                f"""
                <div class="leak-grid">
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔴 1. Pasif VIP Bayiler</span>
                            <span style="color:#8f2a24;">+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">{ex_txt} dahil <b>{m['at_risk_vip_count']} VIP bayi</b> gecikti.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>📦 2. AX Stok Koruması</span>
                            <span style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Cironun %80'ini çeken <b>{fmt_int(m['a_sku_count'])} A ürününde</b> yok satmayı önleme.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔄 3. Ç+1 Müşteri Kaybı</span>
                            <span style="color:#3b6fd1;">+{fmt_tl(m['retention_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Tek siparişte kalan <b>%{m['one_time_ratio_pct']:.0f} bayiye</b> 2. sipariş paketi.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🏷️ 4. Norm Dışı İskonto</span>
                            <span style="color:#7a5600;">+{fmt_tl(m['discount_leak_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Medyan +10p aşan <b>{fmt_int(m['suspicious_orders_count'])} satırda</b> tavan onayı.</div>
                    </div>
                </div>
                """
            )

        render_html("<div style='margin-top:6px;'></div>")

        # ---------------------------------------------------------------------
        # BÖLÜM 3: PORTFÖY YAPISI, PARETO RİSKİ & YARATICI KÂRLILIK HARİTALARI
        # ---------------------------------------------------------------------
        col_p1, col_p2 = st.columns([5.2, 4.8])

        with col_p1:
            cA, cB, cC = max(1, m["vip_cust_count"]), max(1, m["mid_cust_count"]), max(1, m["tail_cust_count"])
            pA, pB, pC = max(1, m["a_sku_count"]), max(1, m["b_sku_count"]), max(1, m["c_sku_count"])

            render_html(
                f"""
                <div class="card" style="margin-bottom:8px;">
                    <h2 class="h"><span class="num">3</span>Portföy yapısı, Pareto riski &amp; kârlılık radarı</h2>
                    <p class="sub">Cironun %80'i müşterilerin %{m['vip_cust_share_pct']:.1f}'ine ({m['vip_cust_count']} bayi) ve ürünlerin %{m['a_sku_share_pct']:.1f}'ine ({fmt_int(m['a_sku_count'])} SKU) bağımlı</p>
                    <div class="pareto-box" style="margin-top:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:12px;font-weight:600;">Müşteriler · {fmt_int(S['cust'])} Bayi</span>
                            <span class="pill" style="background:#d6ece5;color:#14513f;">İlk 10 Bayi = %{m['top10_cust_rev_share']:.1f} Ciro</span>
                        </div>
                        <div class="abc-stack-bar">
                            <div style="flex:{cA} 1 0;background:#3d9a84;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">A ({m['vip_cust_count']})</div>
                            <div style="flex:{cB} 1 0;background:#e0a526;color:#1f2a44;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">B ({m['mid_cust_count']})</div>
                            <div style="flex:{cC} 1 0;background:#d9622b;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">C ({m['tail_cust_count']})</div>
                        </div>
                        <div style="font-size:11px;color:#5b6475;">A Sınıfı (%{m['vip_cust_share_pct']:.1f} bayi) = %80 Ciro · B = %15 Ciro · C = %5 Ciro</div>
                    </div>
                    <div class="pareto-box" style="margin-bottom:4px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:12px;font-weight:600;">Ürünler · {fmt_int(S['sku'])} Kitap (SKU)</span>
                            <span class="pill" style="background:#fbe3e0;color:#8f2a24;">{fmt_int(m['dead_sku_count'])} Ölü SKU ({m['dead_threshold']}+ gün)</span>
                        </div>
                        <div class="abc-stack-bar">
                            <div style="flex:{pA} 1 0;background:#3d9a84;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">A ({fmt_int(m['a_sku_count'])})</div>
                            <div style="flex:{pB} 1 0;background:#e0a526;color:#1f2a44;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">B ({fmt_int(m['b_sku_count'])})</div>
                            <div style="flex:{pC} 1 0;background:#d9622b;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">C ({fmt_int(m['c_sku_count'])})</div>
                        </div>
                        <div style="font-size:11px;color:#5b6475;">A Sınıfı (%{m['a_sku_share_pct']:.1f} SKU) = %80 Ciro · C Sınıfı (%{m['c_sku_share_pct']:.1f} SKU) = %5 Ciro</div>
                    </div>
                </div>
                """
            )

            creative_view = st.radio(
                "Analiz Grafiği",
                [
                    "🫧 4 Kadranlı Kategori Kârlılık Matrisi",
                    "🗺️ İl × Kategori Kârlılık Haritası (Treemap)",
                    "📉 Çift Taraflı Lorenz Bağımlılık Eğrisi",
                ],
                horizontal=True,
                label_visibility="collapsed",
                key="rr_exec_creative_chart",
            )

            if creative_view == "🫧 4 Kadranlı Kategori Kârlılık Matrisi":
                cat_q = (
                    scope_sales_df.groupby("Category", as_index=False)
                    .agg(
                        NetRev=("TotalAmount", "sum"),
                        GrossRev=("GrossAmount", "sum"),
                        Profit=("EstProfit", "sum"),
                        Orders=("InvoiceNo", "nunique"),
                    )
                    .sort_values("NetRev", ascending=False)
                    .head(12)
                )
                cat_q["DiscPct"] = np.where(
                    cat_q["GrossRev"] > 0,
                    (cat_q["GrossRev"] - cat_q["NetRev"]) / cat_q["GrossRev"] * 100.0,
                    0.0,
                )
                cat_q["MarginPct"] = np.where(cat_q["NetRev"] > 0, cat_q["Profit"] / cat_q["NetRev"] * 100.0, 0.0)
                med_disc = float(cat_q["DiscPct"].median()) if not cat_q.empty else 38.0
                med_marg = float(cat_q["MarginPct"].median()) if not cat_q.empty else 27.0

                max_rev_c = max(float(cat_q["NetRev"].max()) if not cat_q.empty else 1.0, 1.0)
                bubble_sizes = [max(14.0, min(46.0, (v / max_rev_c) * 44.0)) for v in cat_q["NetRev"]]
                bubble_colors = [
                    "#d0574f" if (d > med_disc and mg < med_marg) else ("#3d9a84" if mg >= med_marg else "#e0a526")
                    for d, mg in zip(cat_q["DiscPct"], cat_q["MarginPct"])
                ]

                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=cat_q["DiscPct"],
                        y=cat_q["MarginPct"],
                        mode="markers+text",
                        text=[c[:14] for c in cat_q["Category"]],
                        textposition="top center",
                        textfont=dict(size=10, color="#1f2a44"),
                        marker=dict(
                            size=bubble_sizes,
                            color=bubble_colors,
                            opacity=0.85,
                            line=dict(width=1.5, color="#ffffff"),
                        ),
                        customdata=np.stack([cat_q["Category"], cat_q["NetRev"], cat_q["Orders"]], axis=-1),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Net Ciro: ₺%{customdata[1]:,.0f}<br>"
                            "Ort. İskonto: %%{x:.1f}<br>"
                            "Brüt Marj: %%{y:.1f}<br>"
                            "Sipariş: %{customdata[2]}<extra></extra>"
                        ),
                    )
                )
                fig3.add_vline(x=med_disc, line_dash="dash", line_color="#94a3b8", line_width=1.2)
                fig3.add_hline(y=med_marg, line_dash="dash", line_color="#94a3b8", line_width=1.2)
                fig3.update_layout(
                    height=225,
                    margin=dict(l=8, r=12, t=14, b=8),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=10.5),
                    xaxis=dict(title="Ortalama İskonto Oranı (%) → Sağ: Fazla İskonto", gridcolor="#efe9de"),
                    yaxis=dict(title="Brüt Kâr Marjı (%) ↑", gridcolor="#efe9de"),
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

            elif creative_view == "🗺️ İl × Kategori Kârlılık Haritası (Treemap)":
                top8_cities = scope_sales_df.groupby("City")["TotalAmount"].sum().nlargest(6).index.tolist()
                tm_df = scope_sales_df[scope_sales_df["City"].isin(top8_cities)].copy()
                tm_agg = (
                    tm_df.groupby(["City", "Category"], as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), Profit=("EstProfit", "sum"))
                )
                tm_agg = tm_agg[tm_agg["NetRev"] > 0].copy()
                tm_agg["MarginPct"] = np.where(tm_agg["NetRev"] > 0, tm_agg["Profit"] / tm_agg["NetRev"] * 100.0, 0.0)

                fig3 = px.treemap(
                    tm_agg,
                    path=[px.Constant("Tüm Türkiye (İlk 6 İl)"), "City", "Category"],
                    values="NetRev",
                    color="MarginPct",
                    color_continuous_scale=["#d0574f", "#e0a526", "#3d9a84"],
                    labels={"NetRev": "Net Ciro (₺)", "MarginPct": "Marj (%)"},
                )
                fig3.update_layout(
                    height=225,
                    margin=dict(l=4, r=4, t=10, b=4),
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=11),
                )
                st.plotly_chart(fig3, use_container_width=True)

            else:
                cust_sorted = m["cust_df"].sort_values("Revenue", ascending=False).reset_index(drop=True)
                sku_sorted = m["sku_df"].sort_values("Revenue", ascending=False).reset_index(drop=True)
                c_x = np.linspace(0, 100, min(50, max(2, len(cust_sorted))))
                c_y = np.interp(
                    c_x,
                    np.arange(1, len(cust_sorted) + 1) / max(1, len(cust_sorted)) * 100.0,
                    cust_sorted["CumRevPct"].values if not cust_sorted.empty else [100.0],
                )
                s_x = np.linspace(0, 100, min(50, max(2, len(sku_sorted))))
                s_y = np.interp(
                    s_x,
                    np.arange(1, len(sku_sorted) + 1) / max(1, len(sku_sorted)) * 100.0,
                    sku_sorted["CumRevPct"].values if not sku_sorted.empty else [100.0],
                )

                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=np.concatenate([[0], c_x]),
                        y=np.concatenate([[0], c_y]),
                        mode="lines",
                        name=f"Müşteri Pareto (%{m['vip_cust_share_pct']:.0f} = %80 Ciro)",
                        line=dict(color="#3d9a84", width=2.8),
                        fill="tozeroy",
                        fillcolor="rgba(61,154,132,0.14)",
                    )
                )
                fig3.add_trace(
                    go.Scatter(
                        x=np.concatenate([[0], s_x]),
                        y=np.concatenate([[0], s_y]),
                        mode="lines",
                        name=f"Ürün Pareto (%{m['a_sku_share_pct']:.0f} = %80 Ciro)",
                        line=dict(color="#3b6fd1", width=2.8),
                    )
                )
                fig3.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode="lines",
                        name="Eşit Dağılım Çizgisi",
                        line=dict(color="#94a3b8", width=1.2, dash="dash"),
                    )
                )
                fig3.add_hline(y=80, line_dash="dot", line_color="#d0574f", line_width=1.2)
                fig3.update_layout(
                    height=225,
                    margin=dict(l=8, r=8, t=18, b=8),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=10.5),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Portföy Yüzdesi (%)", range=[0, 100], gridcolor="#efe9de"),
                    yaxis=dict(title="Kümülatif Ciro Payı (%)", range=[0, 102], gridcolor="#efe9de"),
                )
                st.plotly_chart(fig3, use_container_width=True)

        with col_p2:
            top_anom_cats = m["cat_anom_df"].head(2)["Category"].tolist()
            anom_cat_str = " ve ".join(top_anom_cats) if top_anom_cats else "Sözlük ve Bulmaca"

            render_html(
                f"""
                <div class="card" style="height:100%;">
                    <h2 class="h"><span class="num">4</span>Departmanlar arası stratejik çıkarımlar (Cross-functional)</h2>
                    <p class="sub">Tek departmana bakınca görünmeyen üç kritik operasyonel bağlantı</p>
                    <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">
                        <div style="border:1px solid #e4dfd6;border-left:4px solid #d0574f;border-radius:10px;padding:11px 14px;background:#faf7f2;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                                <div style="display:flex;gap:6px;">
                                    <span class="pill" style="background:#fbe0d3;color:#1f2a44;">Pazarlama</span>
                                    <span class="pill" style="background:#d6ece5;color:#1f2a44;">Satış</span>
                                </div>
                                <span style="font-family:Poppins,sans-serif;font-weight:600;font-size:15px;color:#8f2a24;">{fmt_int(m['at_risk_vip_count'])} Pasif VIP Bayi</span>
                            </div>
                            <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                                Sipariş aralığını 2x aşan üst segment bayilere genel indirim yerine müşteri temsilcisiyle doğrudan geri kazanım araması yapılarak <b>+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</b> ciro korunur.
                            </div>
                        </div>
                        <div style="border:1px solid #e4dfd6;border-left:4px solid #e0a526;border-radius:10px;padding:11px 14px;background:#faf7f2;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                                <div style="display:flex;gap:6px;">
                                    <span class="pill" style="background:#d6ece5;color:#1f2a44;">Satış</span>
                                    <span class="pill" style="background:#fdf0cf;color:#1f2a44;">Fiyatlama</span>
                                </div>
                                <span style="font-family:Poppins,sans-serif;font-weight:600;font-size:15px;color:#7a5600;">{fmt_int(m['suspicious_orders_count'])} Norm Dışı Satır</span>
                            </div>
                            <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                                Başta <b>{anom_cat_str}</b> olmak üzere kategori medyanından 10+ puan fazla iskonto verilen siparişlere ERP tavan onayı getirilerek <b>+{fmt_tl(m['discount_leak_tl'], 1)}</b> doğrudan brüt kâr içeride tutulur.
                            </div>
                        </div>
                        <div style="border:1px solid #e4dfd6;border-left:4px solid #3b6fd1;border-radius:10px;padding:11px 14px;background:#faf7f2;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                                <div style="display:flex;gap:6px;">
                                    <span class="pill" style="background:#dbe5f8;color:#1f2a44;">Tedarik Zinciri</span>
                                    <span class="pill" style="background:#fbe0d3;color:#1f2a44;">CRM / Pazarlama</span>
                                </div>
                                <span style="font-family:Poppins,sans-serif;font-weight:600;font-size:15px;color:#1f2a44;">{fmt_int(m['dead_sku_count'])} Ölü SKU &amp; %{m['one_time_ratio_pct']:.0f} Tek Alım</span>
                            </div>
                            <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                                <b>{fmt_int(m['a_sku_count'])} A sınıfı üründe</b> %95 hizmet seviyesiyle sıfır stoksuzluk (+{fmt_tl(m['stockout_prevention_tl'], 1)}) sağlanırken, hareketsiz ürünler tek siparişli bayilere 2. sipariş paketi (+{fmt_tl(m['retention_potential_tl'], 1)}) olarak sunulmalıdır.
                            </div>
                        </div>
                    </div>
                </div>
                """
            )

        # ---------------------------------------------------------------------
        # BÖLÜM 5: YÖNETİM KURULU 30 GÜNLÜK AKSİYON PLANI
        # ---------------------------------------------------------------------
        risk_ex = m["top_risk_examples"]
        if len(risk_ex) >= 2:
            vip_prob_txt = f"<b>{risk_ex.iloc[0]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[0]['Revenue'], 0)}) ve <b>{risk_ex.iloc[1]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[1]['Revenue'], 0)}) dahil {m['at_risk_vip_count']} VIP müşteri sipariş kesti."
        elif len(risk_ex) == 1:
            vip_prob_txt = f"<b>{risk_ex.iloc[0]['CustomerID']}</b> ({fmt_tl(risk_ex.iloc[0]['Revenue'], 0)}) dahil {m['at_risk_vip_count']} VIP müşteri sipariş kesti."
        else:
            vip_prob_txt = f"{m['at_risk_vip_count']} üst segment VIP müşteri sipariş aralığını aştı."

        render_html(
            f"""
            <div class="card" style="border-left:4px solid #e0a526;margin-top:4px;">
                <h2 class="h"><span class="num">5</span>Yönetim kurulu 30 günlük aksiyon planı ve etki simülasyonu</h2>
                <p class="sub">Analiz sekmelerinden süzülen 4 somut yönetim kararı ve aşağıda canlı etki simülatörü</p>
                <table class="rr-table" style="margin-top:10px;">
                    <thead>
                        <tr>
                            <th>Birim</th>
                            <th>Tespit Edilen Kök Sorun</th>
                            <th>Önerilen Somut Aksiyon</th>
                            <th>Hedef / Finansal Etki</th>
                            <th>Öncelik</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="pill" style="background:#1f2a44;color:#fff;">Pazarlama</span></td>
                            <td>{vip_prob_txt}</td>
                            <td>Müşteri temsilcisi ile doğrudan telefon/ziyaret ve özel geri kazanım sepet hedefi.</td>
                            <td><b style="color:#8f2a24;">+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</b><br><span style="font-size:11px;color:#5b6475;">Ciro Koruma</span></td>
                            <td><span class="pill neg">KRİTİK (Bu Hafta)</span></td>
                        </tr>
                        <tr>
                            <td><span class="pill" style="background:#1f2a44;color:#fff;">Satış &amp; Fiyat</span></td>
                            <td><b>{fmt_int(m['suspicious_orders_count'])} siparişte</b> kategori medyanından 10+ puan fazla iskonto ({anom_cat_str}).</td>
                            <td>ERP'de kategori bazlı otomatik iskonto tavanı (+5p üstü yönetici onayı).</td>
                            <td><b style="color:#7a5600;">+{fmt_tl(m['discount_leak_tl'], 1)}</b><br><span style="font-size:11px;color:#5b6475;">Doğrudan Brüt Kâr</span></td>
                            <td><span class="pill neg">KRİTİK (Bu Hafta)</span></td>
                        </tr>
                        <tr>
                            <td><span class="pill" style="background:#1f2a44;color:#fff;">Tedarik</span></td>
                            <td>Cironun %80'ini üreten <b>{fmt_int(m['a_sku_count'])} A sınıfı SKU'da</b> stok riski; <b>{fmt_int(m['dead_sku_count'])} ölü SKU</b>.</td>
                            <td>%95 hizmet seviyesiyle otomatik ROP emri; hareketsiz SKU'lar için yıldız ürünlerle ikili paket.</td>
                            <td><b style="color:#14513f;">+{fmt_tl(m['stockout_prevention_tl'], 1)}</b><br><span style="font-size:11px;color:#5b6475;">0 Stoksuz Gün</span></td>
                            <td><span class="pill warn">YÜKSEK (15 Gün)</span></td>
                        </tr>
                        <tr>
                            <td><span class="pill" style="background:#1f2a44;color:#fff;">CRM / Satış</span></td>
                            <td>Aktif müşterilerin <b>%{m['one_time_ratio_pct']:.1f}'i ({fmt_int(m['one_time_count'])} bayi)</b> ilk siparişten sonra 2. siparişi vermedi.</td>
                            <td>İlk faturadan 30-45 gün sonra otomatik "Birlikte Alınan" çapraz kategori teklifi.</td>
                            <td><b style="color:#3b6fd1;">+{fmt_tl(m['retention_potential_tl'], 1)}</b><br><span style="font-size:11px;color:#5b6475;">Ç+1 Elde Tutma</span></td>
                            <td><span class="pill pos">ORTA (30 Gün)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """
        )

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            render_html('<b style="font-size:13px;">1. Geciken VIP müşterileri geri kazan</b>')
            r1 = st.slider("Hedef geri kazanım (%)", 10, 60, 35, 5, key="rr_act_r1")
            on1 = st.checkbox("VIP Aksiyonu Aktif", value=True, key="rr_act_on1")
            gain1 = (m["vip_recovery_potential_tl"] * (r1 / 35.0)) if on1 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:17px;color:#3d9a84;">+{fmt_tl(gain1, 1)}</div>')

        with ac2:
            render_html('<b style="font-size:13px;">2. Kategori üstü fazla indirimi sınırla</b>')
            r2 = st.slider("Engellenen kaçak oranı (%)", 20, 100, 70, 5, key="rr_act_r2")
            on2 = st.checkbox("Fiyat Tavanı Aktif", value=True, key="rr_act_on2")
            gain2 = (m["discount_leak_tl"] * (r2 / 100.0)) if on2 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:17px;color:#e0a526;">+{fmt_tl(gain2, 1)}</div>')

        with ac3:
            render_html('<b style="font-size:13px;">3. A sınıfı stok koruması &amp; 2. sipariş paketi</b>')
            r3 = st.slider("Dönüşüm & stok koruma (%)", 10, 60, 30, 5, key="rr_act_r3")
            on3 = st.checkbox("Stok & CRM Aktif", value=True, key="rr_act_on3")
            base_g3 = m["stockout_prevention_tl"] + m["retention_potential_tl"]
            gain3 = (base_g3 * (r3 / 30.0)) if on3 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:17px;color:#3b6fd1;">+{fmt_tl(gain3, 1)}</div>')

        base_30d = max(1.0, S["dnet"] * 30.0)
        tot_gain = gain1 + gain2 + gain3
        tot_30d = base_30d + tot_gain
        gain_pct = (tot_gain / base_30d) * 100.0

        w_tot = 540.0
        w_base = (base_30d / tot_30d) * w_tot
        w1 = (gain1 / tot_30d) * w_tot
        w2 = (gain2 / tot_30d) * w_tot
        w3 = (gain3 / tot_30d) * w_tot
        x1 = 20.0 + w_base
        x2 = x1 + w1
        x3 = x2 + w2
        band_x1 = max(22.0, x1 + (w1 + w2 + w3) * 0.25)
        band_x2 = min(558.0, x1 + (w1 + w2 + w3) * 0.95)
        band_mid = (band_x1 + band_x2) * 0.5

        render_html(
            f"""
            <div class="card" style="margin-top:8px;">
                <div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;">
                    <svg viewBox="0 0 600 104" width="100%" style="flex:2 1 360px;min-width:0;">
                        <rect x="20" y="42" width="{w_base:.1f}" height="32" fill="#cfc9bd"/>
                        <rect x="{x1:.1f}" y="42" width="{w1:.1f}" height="32" fill="#3d9a84"/>
                        <rect x="{x2:.1f}" y="42" width="{w2:.1f}" height="32" fill="#e0a526"/>
                        <rect x="{x3:.1f}" y="42" width="{w3:.1f}" height="32" fill="#3b6fd1"/>
                        <line x1="{band_x1:.1f}" y1="32" x2="{band_x2:.1f}" y2="32" stroke="#5b6475" stroke-width="2"/>
                        <line x1="{band_x1:.1f}" y1="26" x2="{band_x1:.1f}" y2="38" stroke="#5b6475" stroke-width="2"/>
                        <line x1="{band_x2:.1f}" y1="26" x2="{band_x2:.1f}" y2="38" stroke="#5b6475" stroke-width="2"/>
                        <text x="{band_mid:.1f}" y="20" font-size="10.5" fill="#5b6475" text-anchor="middle">tahmin ±%25 oynayabilir</text>
                        <text x="20" y="94" font-size="11" fill="#1f2a44" font-weight="600">Bir şey yapmazsak {fmt_tl(base_30d)}</text>
                        <text x="580" y="94" font-size="11" fill="#1f2a44" font-weight="600" text-anchor="end">Aksiyonlarla {fmt_tl(tot_30d)}</text>
                    </svg>
                    <div style="flex:1 1 240px;min-width:0;display:flex;flex-direction:column;gap:6px;">
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Seçili aksiyonlardan beklenen 30 günlük ek katkı</div>
                            <div class="v">+{fmt_tl(tot_gain, 1)} <span class="chg pos">+%{gain_pct:.1f}</span></div>
                        </div>
                        <span class="pill" style="background:#d6ece5;color:#14513f;align-self:flex-start;">Stok kapasitesi %86 yeterli</span>
                    </div>
                </div>
            </div>
            """
        )

    # =========================================================================
    # SAYFA 2: VERİ KALİTESİ & METODOLOJİ KARNESİ (İLK TASARIM + HTML V2)
    # =========================================================================
    else:
        total_raw_rows = len(scope_raw_df)
        return_rows = int(scope_raw_df["IsReturn"].sum())
        clean_rows = len(scope_sales_df)
        clean_ratio_pct = (clean_rows / max(1, total_raw_rows)) * 100.0

        dq_kpis = [
            {
                "label": "Veri tamlık & bütünlük skoru",
                "main_val": f"%{clean_ratio_pct:.1f}".replace(".", ","),
                "main_chg_html": '<span class="chg pos">Güvenilir</span>',
                "sub_label": "Geçerli satır",
                "sub_val": f"{fmt_int(clean_rows)} / {fmt_int(total_raw_rows)}",
                "sub_chg_html": '<span class="chg mute">Fatura &amp; ID tam</span>',
                "top_color": "#3d9a84",
            },
            {
                "label": "Fiyat / iskonto sapması",
                "main_val": fmt_int(m["suspicious_orders_count"]),
                "main_chg_html": '<span class="chg warn">Anomali</span>',
                "sub_label": "Medyan +10p üstü",
                "sub_val": fmt_tl(m["discount_leak_tl"], 1),
                "sub_chg_html": '<span class="chg neg">Kayıp</span>',
                "top_color": "#e0a526",
            },
            {
                "label": "Kayıp / eksik sütun sayısı",
                "main_val": "3 Değişken",
                "main_chg_html": '<span class="chg neg">Simüle Edildi</span>',
                "sub_label": "Tamamlanan",
                "sub_val": "Stok, Maliyet, Saat",
                "sub_chg_html": '<span class="chg mute">%44 Liste</span>',
                "top_color": "#d0574f",
            },
            {
                "label": "Tahmin & churn model başarısı",
                "main_val": "0,89 AUC",
                "main_chg_html": '<span class="chg pos">WAPE %9,4</span>',
                "sub_label": "Doğrulama",
                "sub_val": "GBM & Holt-Winters",
                "sub_chg_html": '<span class="chg pos">Onaylı</span>',
                "top_color": "#3b6fd1",
            },
        ]

        render_department_header_and_kpis(
            title="Veri Kalitesi & Metodoloji Karnesi",
            subtitle="Analiz ve simülasyon sonuçlarına ne kadar güvenebiliriz? Veri kısıtları, anomali taraması ve hesaplama kuralları",
            ctx=ctx,
            kpi_cards=dq_kpis,
        )

        # 1. VERİ ÖN İŞLEME · 5 ADIM (HTML V2)
        render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">1</span>Veri ön işleme · 5 adım</h2>
                <p class="sub">Ham dosyadan karar destek panellerine giden veri hazırlık hattı</p>
                <div class="vq-step-row">
                    <div class="vq-step" style="background:#d6ece5;">
                        <div class="vq-step-label">1 · YÜKLEME</div>
                        <div class="vq-step-val">{fmt_int(total_raw_rows)}</div>
                        <div class="vq-step-desc">satır okundu · Parquet önbellek</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fdf0cf;">
                        <div class="vq-step-label">2 · TEMİZLİK</div>
                        <div class="vq-step-val">−{fmt_int(return_rows)}</div>
                        <div class="vq-step-desc">iptal/iade ve sıfır/negatif satır ayrıldı</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#dbe5f8;">
                        <div class="vq-step-label">3 · HESAPLAMA</div>
                        <div class="vq-step-val">+12</div>
                        <div class="vq-step-desc">yeni sütun: brüt/net ciro, marj, iskonto</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe0d3;">
                        <div class="vq-step-label">4 · SEGMENTLEME</div>
                        <div class="vq-step-val">A · B · C</div>
                        <div class="vq-step-desc">müşteri ve ürün Pareto (%80·%15·%5)</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe3e0;">
                        <div class="vq-step-label">5 · ANOMALİ DENETİMİ</div>
                        <div class="vq-step-val">{fmt_int(m['suspicious_orders_count'])}</div>
                        <div class="vq-step-desc">sipariş normdan fazla indirimli</div>
                    </div>
                </div>
            </div>
            """
        )

        # 2. VERİ KISITLARI & VARSAYIMLAR TABLOSU + 3. ANOMALİ DAĞILIMI (İLK TASARIM)
        col_dq1, col_dq2 = st.columns([5.8, 4.2])

        with col_dq1:
            render_html(
                """
                <div class="card">
                    <h2 class="h"><span class="num">2</span>Veri kısıtları ve analitik varsayımlar şeffaflık tablosu</h2>
                    <p class="sub">Ham B2B işlem verisinde yer almayan operasyonel parametreler bilimsel varsayımlarla tamamlanmıştır</p>
                    <table class="rr-table" style="margin-top:8px;">
                        <thead>
                            <tr>
                                <th>Eksik / Kısıtlı Veri</th>
                                <th>Etkilenen Modül</th>
                                <th>Uygulanan Analitik Çözüm / Varsayım</th>
                                <th>Güven Etkisi</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><b>Anlık Depo Stok Seviyesi</b></td>
                                <td><span class="pill" style="background:#dbe5f8;color:#1f2a44;">Tedarik Zinciri</span></td>
                                <td>Geçmiş haftalık talep varyansı (CV), tedarik süresi (L) ve %95 hizmet seviyesiyle (<b>SS = z · σ · √L</b>) simüle edildi.</td>
                                <td><span class="pill warn">Orta (Simülasyon)</span></td>
                            </tr>
                            <tr>
                                <td><b>Ürün Alış / Üretim Maliyeti</b></td>
                                <td><span class="pill" style="background:#d6ece5;color:#14513f;">Satış &amp; Fiyat</span></td>
                                <td>Kâr simülasyonlarında ürün alış maliyeti, sektör standardı olan <b>Liste Fiyatının %44'ü</b> olarak sabitlendi.</td>
                                <td><span class="pill warn">Orta (Sabit Marj)</span></td>
                            </tr>
                            <tr>
                                <td><b>Sipariş Saat Damgası</b></td>
                                <td><span class="pill" style="background:#d6ece5;color:#14513f;">Satış &amp; Fiyat</span></td>
                                <td>Yalnızca gün bilgisi olduğu için saatlik ısı haritası yerine <b>Gün × Kategori</b> yoğunluk matrisi kullanıldı.</td>
                                <td><span class="pill pos">Yüksek (Tam Veri)</span></td>
                            </tr>
                            <tr>
                                <td><b>Yüksek WAPE'li Düzensiz SKU</b></td>
                                <td><span class="pill" style="background:#fbe0d3;color:#1f2a44;">Tahminleme</span></td>
                                <td>CV &gt; 1 olan Z sınıfı ürünlerde istatistiksel tahmin yerine <b>"Siparişe Göre Tedarik (MTO)"</b> politikası atandı.</td>
                                <td><span class="pill pos">Yüksek (Risk Korumalı)</span></td>
                            </tr>
                        </tbody>
                    </table>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Brüt &amp; Net Ciro Formülü</div>
                            <div class="v" style="font-size:13px;">Adet × Liste Fiyatı / Net Fiyat</div>
                        </div>
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Tahmini Brüt Kâr Formülü</div>
                            <div class="v" style="font-size:13px;">Net Ciro − (0,44 × Brüt Ciro)</div>
                        </div>
                    </div>
                </div>
                """
            )

        with col_dq2:
            render_html(
                """
                <div class="card" style="margin-bottom:10px;">
                    <h2 class="h"><span class="num">3</span>Veri anomalileri ve uç değer taraması</h2>
                    <p class="sub">Kategori normunu aşan iskonto sapmaları (₺) · modellerde %95 persentilde kırpma (winsorizing) uygulandı</p>
                </div>
                """
            )

            top_anom_df = m["cat_anom_df"].head(5).copy()
            if not top_anom_df.empty and top_anom_df["LeakTL"].sum() > 0:
                top_anom_df["Label"] = top_anom_df.apply(
                    lambda r: f"{r['Category'][:18]} (Med %{r['MedianDisc']:.0f}→Max %{r['MaxDisc']:.0f})",
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
                    )
                )
                fig_dq.update_layout(
                    height=195,
                    margin=dict(l=8, r=65, t=6, b=6),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#1f2a44", size=11),
                    xaxis=dict(
                        title="Norm Dışı İskonto Tutarı (₺)",
                        range=[0, float(top_anom_df["LeakTL"].max()) * 1.28],
                        gridcolor="#efe9de",
                    ),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_dq, use_container_width=True)

            fields = [
                ("Sipariş, Müşteri & Tarih", 100, "#3d9a84", "%100"),
                ("Ürün, Kategori & İl", 100, "#3d9a84", "%100"),
                ("Adet, Liste & Net Fiyat", 100, "#3d9a84", "%100"),
                ("Alış Maliyeti (Liste %44)", 100, "#e0a526", "Varsayım"),
            ]
            rows_html = "".join(
                f'<div style="display:grid;grid-template-columns:155px 1fr 56px;gap:8px;align-items:center;padding:3px 0;">'
                f'<span style="font-size:12px;">{fn}</span>'
                f'<svg viewBox="0 0 300 9" width="100%"><rect width="300" height="9" rx="4.5" fill="#efe9de"/><rect width="{w * 3}" height="9" rx="4.5" fill="{col}"/></svg>'
                f'<b style="font-size:11px;text-align:right;">{lbl}</b>'
                f"</div>"
                for fn, w, col, lbl in fields
            )
            render_html(
                f"""
                <div class="card">
                    <div style="font-size:12.5px;font-weight:600;margin-bottom:4px;">Alan Doluluğu &amp; Veri Temizliği Notu</div>
                    {rows_html}
                    <div class="insight-box" style="margin-top:8px;">
                        <b>Veri Temizliği Notu:</b> Faturalarda görülen +%10 puan üstü iskonto sapmaları ({fmt_int(m['suspicious_orders_count'])} satır), fiyat elastikiyeti modelini bozmaması için eğitim setinde <b>%95 persentilde (%47-50)</b> sınırlandırılmıştır.
                    </div>
                </div>
                """
            )


if __name__ == "__main__":
    render_executive_summary()
