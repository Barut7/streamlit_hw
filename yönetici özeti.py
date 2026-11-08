import os
import datetime
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI (DOĞRUDAN ÇALIŞTIRILDIĞINDA)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RafRadar Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_html(html_str: str):
    """Girintileri temizleyerek Markdown'ın HTML'i kod bloğu sanmasını engeller."""
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
        .track {
            display: grid;
            grid-template-columns: 195px minmax(140px, 1fr) 170px;
            gap: 12px;
            align-items: center;
            padding: 7px 0;
            border-bottom: 1px solid #f0ebe2;
        }
        .hm { width: 100%; border-collapse: separate; border-spacing: 0; }
        .hm td { text-align: center; font-size: 12px; font-weight: 600; padding: 7px 4px; border: 2px solid #fff; }
        .hm th { font-size: 11px; font-weight: 600; color: #5b6475; padding: 4px; text-align: center; border: none; }
        .rr-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .rr-table th { font-size: 11px; font-weight: 600; color: #5b6475; text-align: left; padding: 6px 8px; border-bottom: 1px solid #e4dfd6; }
        .rr-table td { font-size: 12px; padding: 8px; border-bottom: 1px solid #f0ebe2; vertical-align: middle; }
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
    take_html = f'<div style="font-size:13.5px;font-weight:600;margin:6px 0 0 0;line-height:1.4;color:#1f2a44;">{takeaway}</div>' if takeaway else ""
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
    opts = pages if pages is not None else (["Yönetici Özeti", "Veri Kalitesi"] if show_page_nav else ["Yönetici Özeti"])
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
            f"tahmini kâr **{fmt_tl(m['est_profit'])} (%{m['profit_margin']:.1f} marj)**."
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
            f"Müşterilerin **%{m['vip_cust_share_pct']:.1f}'i** ve ürünlerin **%{m['a_sku_share_pct']:.1f}'i** cironun %80'ini getiriyor."
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
            f"Stok & Paket +{fmt_tl(m['stockout_prevention_tl'] + m['retention_potential_tl'])})."
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
# 7. YÖNETİCİ ÖZETİ & VERİ KALİTESİ (RAFRADAR-VERSIYON2.HTML BİREBİR UYARLAMA)
# -----------------------------------------------------------------------------
def _build_track_svg(monthly_vals: list, yr_mean: float, sel_val: float, color: str, fmt_fn) -> str:
    vals = [float(v) for v in monthly_vals if not pd.isna(v)] + [float(yr_mean), float(sel_val)]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    span = max(1e-6, hi - lo)

    def _x(v):
        return 14.0 + ((float(v) - lo) / span) * 272.0

    dots_svg = "".join(
        f'<circle cx="{_x(v):.1f}" cy="18" r="3.5" fill="#b9b3a6"><title>{MONTH_TR_SHORT[i]}: {fmt_fn(v)}</title></circle>'
        for i, v in enumerate(monthly_vals)
    )
    mx = _x(yr_mean)
    sx = _x(sel_val)
    return (
        f'<svg viewBox="0 0 300 40" width="100%">'
        f'<rect x="8" y="14" width="284" height="8" rx="4" fill="#efe9de"/>'
        f"{dots_svg}"
        f'<line x1="{mx:.1f}" y1="7" x2="{mx:.1f}" y2="29" stroke="#1f2a44" stroke-width="2"/>'
        f'<circle cx="{sx:.1f}" cy="18" r="7.5" fill="{color}" stroke="#fff" stroke-width="2"><title>Seçili: {fmt_fn(sel_val)}</title></circle>'
        f'<text x="8" y="38" font-size="9" fill="#5b6475">{fmt_fn(lo)}</text>'
        f'<text x="292" y="38" font-size="9" fill="#5b6475" text-anchor="end">{fmt_fn(hi)}</text>'
        f"</svg>"
    )


def _build_yearly_overview_svg(monthly_df: pd.DataFrame) -> str:
    """rafradar-versiyon2.html 1. paneldeki (Yıllık genel bakış) SVG grafik."""
    revs = monthly_df["NetRev"].tolist()
    discs = monthly_df["DiscPct"].tolist()
    max_r = max(max(revs, default=1.0), 1.0) * 1.12
    bars_svg = ""
    pts = []
    dots_svg = ""
    peak_idx = int(np.argmax(revs)) if revs else 0

    for i in range(12):
        r = revs[i] if i < len(revs) else 0.0
        d = discs[i] if i < len(discs) else 0.0
        x = 46 + i * 45
        lx = x + 15
        h = max(2.0, (r / max_r) * 145.0)
        y = 170.0 - h
        fill = "#3d9a84" if i == peak_idx else "#1f2a44"
        bars_svg += (
            f'<rect x="{x}" y="{y:.1f}" width="30" height="{h:.1f}" rx="3" fill="{fill}">'
            f"<title>{MONTH_TR_FULL[i]}: {fmt_tl(r)} · İndirim %{d:.1f}</title></rect>"
            f'<text x="{lx}" y="188" font-size="10" fill="#5b6475" text-anchor="middle">{MONTH_TR_SHORT[i]}</text>'
        )
        dy = 170.0 - (min(max(d, 0.0), 60.0) / 60.0) * 140.0
        pts.append(f"{lx:.1f},{dy:.1f}")
        dots_svg += f'<circle cx="{lx:.1f}" cy="{dy:.1f}" r="3.5" fill="#e0a526" stroke="#fff" stroke-width="1"/>'

    peak_x = 46 + peak_idx * 45 + 15
    peak_txt = f"Zirve: {MONTH_TR_SHORT[peak_idx]} ({fmt_tl(revs[peak_idx], 1)})" if revs else ""
    mid_lbl = fmt_tl(max_r * 0.5, 1)
    top_lbl = fmt_tl(max_r, 1)

    return (
        f'<svg viewBox="0 0 600 205" width="100%" style="margin-top:6px;">'
        f'<line x1="40" y1="170" x2="590" y2="170" stroke="#cfc9bd"/>'
        f'<line x1="40" y1="95" x2="590" y2="95" stroke="#efe9de"/>'
        f'<line x1="40" y1="20" x2="590" y2="20" stroke="#efe9de"/>'
        f'<text x="36" y="174" font-size="9.5" fill="#5b6475" text-anchor="end">0</text>'
        f'<text x="36" y="99" font-size="9.5" fill="#5b6475" text-anchor="end">{mid_lbl}</text>'
        f'<text x="36" y="24" font-size="9.5" fill="#5b6475" text-anchor="end">{top_lbl}</text>'
        f"{bars_svg}"
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#e0a526" stroke-width="2.5"/>'
        f"{dots_svg}"
        f'<text x="{peak_x}" y="13" font-size="10" fill="#1f2a44" text-anchor="middle" font-weight="600">{peak_txt}</text>'
        f"</svg>"
    )


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
    c_days = ctx["c_days"]
    selected_cat = ctx["category"]
    selected_page = ctx["page"]

    is_kalite = "Veri Kalitesi" in str(selected_page)

    # =========================================================================
    # SAYFA 1: YÖNETİCİ ÖZETİ (rafradar-versiyon2.html 'yonetici' Birebir)
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
                "label": "Tahmini brüt kâr",
                "main_val": fmt_tl(S["profit"]),
                "main_chg_html": f'<span class="chg pos">%{S["margin"]:.1f} marj</span>',
                "sub_label": "Günlük ort.",
                "sub_val": fmt_tl(S["dprofit"], 1),
                "sub_chg_html": chg_badge_html(m["d_profit_chg"]),
                "top_color": "#e0a526",
            },
            {
                "label": "Aktif müşteri",
                "main_val": fmt_int(S["cust"]),
                "main_chg_html": chg_badge_html(m["cust_chg"]),
                "sub_label": "Günlük ort.",
                "sub_val": f"{S['dcust']:.1f}".replace(".", ","),
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
                "label": "Sipariş adedi",
                "main_val": fmt_int(S["orders"]),
                "main_chg_html": chg_badge_html(m["inv_chg"]),
                "sub_label": "Günlük ort.",
                "sub_val": f"{S['dorders']:.1f}".replace(".", ","),
                "sub_chg_html": chg_badge_html(m["d_inv_chg"]),
                "top_color": "#1f2a44",
            },
        ]

        render_department_header_and_kpis(
            title="Yönetici Özeti",
            subtitle="Seçtiğim dönem yılın neresinde ve önümüzdeki 30 günde ne yapmalıyız?",
            ctx=ctx,
            kpi_cards=kpi_cards,
        )

        # ---------------------------------------------------------------------
        # BÖLÜM 1 (YILLIK GENEL BAKIŞ) & BÖLÜM 2 (PORTFÖY YAPISI)
        # ---------------------------------------------------------------------
        col_y1, col_y2 = st.columns([6.3, 3.7])

        with col_y1:
            yh1, yh2 = st.columns([7.5, 2.5])
            with yh1:
                render_html(
                    """
                    <div>
                        <h2 class="h"><span class="num">1</span>Yıllık genel bakış</h2>
                        <p class="sub">Tarih filtresinden bağımsız · çubuk: aylık net ciro · çizgi: indirim oranı</p>
                    </div>
                    """
                )
            with yh2:
                yr_sel = st.selectbox("Yıl", ["2020", "2019", "2021"], index=0, label_visibility="collapsed", key="rr_exec_yr")

            if yr_sel != "2020":
                render_html(
                    f"""
                    <div class="card" style="padding:36px 16px;text-align:center;background:#faf7f2;">
                        <b style="color:#1f2a44;">{yr_sel} için veri yok.</b><br>
                        <span style="font-size:12px;color:#5b6475;">Veri seti yalnız 2020 yılını kapsıyor.</span>
                    </div>
                    """
                )
            else:
                m_grid = pd.DataFrame({"Month": list(range(1, 13))})
                m_agg = (
                    valid_all_df.groupby("Month", as_index=False)
                    .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"))
                )
                m_grid = m_grid.merge(m_agg, on="Month", how="left").fillna(0.0)
                m_grid["DiscPct"] = np.where(
                    m_grid["GrossRev"] > 0,
                    (m_grid["GrossRev"] - m_grid["NetRev"]) / m_grid["GrossRev"] * 100.0,
                    0.0,
                )
                yr_tot_net = float(m_grid["NetRev"].sum())
                yr_tot_gross = float(m_grid["GrossRev"].sum())
                yr_avg_disc = ((yr_tot_gross - yr_tot_net) / yr_tot_gross * 100.0) if yr_tot_gross > 0 else 0.0
                svg_yr = _build_yearly_overview_svg(m_grid)

                render_html(
                    f"""
                    <div class="card">
                        {svg_yr}
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">
                            <span class="pill" style="background:#d6ece5;color:#14513f;">Yıllık net ciro {fmt_tl(yr_tot_net)}</span>
                            <span class="pill" style="background:#fdf0cf;color:#7a5600;">İndirim oranı %{yr_avg_disc:.1f}</span>
                            <span class="pill" style="background:#ece7dd;color:#5b6475;">Çubuk üzerine gel: ay detayı görünür</span>
                        </div>
                    </div>
                    """
                )

        with col_y2:
            cA, cB, cC = max(1, m["vip_cust_count"]), max(1, m["mid_cust_count"]), max(1, m["tail_cust_count"])
            pA, pB, pC = max(1, m["a_sku_count"]), max(1, m["b_sku_count"]), max(1, m["c_sku_count"])
            render_html(
                f"""
                <div class="card" style="height:100%;">
                    <h2 class="h"><span class="num">2</span>Portföy yapısı</h2>
                    <p class="sub">Müşteri ve ürünler ciro payına göre A (%80) · B (%15) · C (%5)</p>
                    <div style="margin-top:12px;">
                        <div style="font-size:12px;font-weight:600;">Müşteriler · {fmt_int(S['cust'])}</div>
                        <div style="display:flex;height:28px;border-radius:8px;overflow:hidden;margin-top:5px;">
                            <div style="flex:{cA} 1 0;background:#3d9a84;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;">A</div>
                            <div style="flex:{cB} 1 0;background:#e0a526;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#1f2a44;">B</div>
                            <div style="flex:{cC} 1 0;background:#d9622b;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;">C</div>
                        </div>
                        <div style="font-size:11px;color:#5b6475;margin-top:4px;">
                            A {fmt_int(m['vip_cust_count'])} · B {fmt_int(m['mid_cust_count'])} · C {fmt_int(m['tail_cust_count'])} müşteri
                        </div>
                    </div>
                    <div style="margin-top:12px;">
                        <div style="font-size:12px;font-weight:600;">Ürünler · {fmt_int(S['sku'])}</div>
                        <div style="display:flex;height:28px;border-radius:8px;overflow:hidden;margin-top:5px;">
                            <div style="flex:{pA} 1 0;background:#3d9a84;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;">A</div>
                            <div style="flex:{pB} 1 0;background:#e0a526;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#1f2a44;">B</div>
                            <div style="flex:{pC} 1 0;background:#d9622b;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;">C</div>
                        </div>
                        <div style="font-size:11px;color:#5b6475;margin-top:4px;">
                            A {fmt_int(m['a_sku_count'])} · B {fmt_int(m['b_sku_count'])} · C {fmt_int(m['c_sku_count'])} ürün
                        </div>
                    </div>
                    <div style="margin-top:12px;font-size:12px;line-height:1.4;padding:9px 11px;background:#faf7f2;border-radius:8px;">
                        Müşterilerin <b>%{m['vip_cust_share_pct']:.1f}</b>'i, ürünlerin <b>%{m['a_sku_share_pct']:.1f}</b>'i cironun %80'ini getiriyor.
                    </div>
                </div>
                """
            )

        # ---------------------------------------------------------------------
        # BÖLÜM 3: DÖNEM ÖZETİ (.track SVG PERFORMANS ÇUBUKLARI)
        # ---------------------------------------------------------------------
        m_daily = (
            valid_all_df.groupby("Month")
            .apply(
                lambda g: pd.Series(
                    {
                        "dnet": g["TotalAmount"].sum() / max(1, g["Date"].nunique()),
                        "dord": g["InvoiceNo"].nunique() / max(1, g["Date"].nunique()),
                        "disc": ((g["GrossAmount"].sum() - g["TotalAmount"].sum()) / max(1.0, g["GrossAmount"].sum())) * 100.0,
                        "dcust": g.groupby("Date")["CustomerID"].nunique().mean(),
                        "dqty": g["Quantity"].sum() / max(1, g["Date"].nunique()),
                        "aov": g["TotalAmount"].sum() / max(1, g["InvoiceNo"].nunique()),
                    }
                )
            )
            .reindex(range(1, 13))
            .fillna(0.0)
        )
        yr_days = max(1, valid_all_df["Date"].nunique())
        yr_dnet = float(valid_all_df["TotalAmount"].sum() / yr_days)
        yr_dord = float(valid_all_df["InvoiceNo"].nunique() / yr_days)
        yr_disc = float(((valid_all_df["GrossAmount"].sum() - valid_all_df["TotalAmount"].sum()) / max(1.0, valid_all_df["GrossAmount"].sum())) * 100.0)
        yr_dcust = float(valid_all_df.groupby("Date")["CustomerID"].nunique().mean())
        yr_dqty = float(valid_all_df["Quantity"].sum() / yr_days)
        yr_aov = float(valid_all_df["TotalAmount"].sum() / max(1, valid_all_df["InvoiceNo"].nunique()))

        def _rel_yr(a, b):
            return ((a - b) / abs(b) * 100.0) if b else 0.0

        track_specs = [
            ("Günlük net ciro", "Satış", "#3d9a84", m_daily["dnet"].tolist(), yr_dnet, S["dnet"], lambda v: fmt_tl(v, 0), chg_badge_html(_rel_yr(S["dnet"], yr_dnet))),
            ("Günlük sipariş", "Satış", "#3d9a84", m_daily["dord"].tolist(), yr_dord, S["dorders"], lambda v: f"{v:.0f}", chg_badge_html(_rel_yr(S["dorders"], yr_dord))),
            ("Ortalama indirim", "Fiyatlama", "#e0a526", m_daily["disc"].tolist(), yr_disc, S["disc"], lambda v: f"%{v:.1f}", chg_badge_html(_rel_yr(S["disc"], yr_disc), invert=True)),
            ("Günlük aktif müşteri", "Pazarlama", "#d9622b", m_daily["dcust"].tolist(), yr_dcust, S["dcust"], lambda v: f"{v:.0f}", chg_badge_html(_rel_yr(S["dcust"], yr_dcust))),
            ("Günlük satılan adet", "Tedarik", "#3b6fd1", m_daily["dqty"].tolist(), yr_dqty, S["dqty"], lambda v: fmt_int(v), chg_badge_html(_rel_yr(S["dqty"], yr_dqty))),
            ("Ortalama sepet tutarı", "Satış & Pazarlama", "#1f2a44", m_daily["aov"].tolist(), yr_aov, S["aov"], lambda v: fmt_tl(v, 0), chg_badge_html(_rel_yr(S["aov"], yr_aov))),
        ]

        tracks_html = ""
        for t_name, t_dep, t_col, t_list, t_yr, t_sel, t_fmt, t_badge in track_specs:
            svg_t = _build_track_svg(t_list, t_yr, t_sel, t_col, t_fmt)
            tracks_html += (
                f'<div class="track">'
                f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;">'
                f'<span class="dot" style="background:{t_col};"></span>'
                f'<div><b>{t_name}</b><div style="font-size:11px;color:#5b6475;">{t_dep}</div></div>'
                f"</div>"
                f"{svg_t}"
                f'<div style="text-align:right;">'
                f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:15px;">{t_fmt(t_sel)}</div>'
                f"{t_badge}"
                f"</div>"
                f"</div>"
            )

        render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">3</span>Dönem özeti</h2>
                <p class="sub">Seçili dönem ({sel_lbl}) günlük ortalamalarının 12 ay ve yıl ortalamasına göre konumu</p>
                <div style="margin-top:6px;">
                    {tracks_html}
                </div>
                <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:#5b6475;margin-top:8px;align-items:center;">
                    <span><svg width="12" height="12" viewBox="0 0 14 14"><circle cx="7" cy="7" r="6" fill="#3d9a84"/></svg> seçili dönem</span>
                    <span><svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="3.5" fill="#b9b3a6"/></svg> 12 ayın günlük ortalaması</span>
                    <span><svg width="10" height="12" viewBox="0 0 10 14"><line x1="5" y1="1" x2="5" y2="13" stroke="#1f2a44" stroke-width="2"/></svg> yıl ortalaması</span>
                </div>
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # BÖLÜM 4: CROSS-FUNCTIONAL INSIGHTS (DEPARTMANLAR ARASI ÇIKARIMLAR)
        # ---------------------------------------------------------------------
        render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">4</span>Cross-functional insights · departmanlar arası çıkarımlar</h2>
                <p class="sub">Tek departmana bakınca görünmeyen üç kritik bağlantı</p>
                <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;">
                    <div style="flex:1 1 250px;min-width:0;border:1px solid #e4dfd6;border-radius:10px;padding:12px 14px;background:#faf7f2;display:flex;flex-direction:column;gap:5px;">
                        <div style="display:flex;gap:6px;flex-wrap:wrap;">
                            <span class="pill" style="background:#fbe0d3;color:#1f2a44;">Pazarlama</span>
                            <span class="pill" style="background:#d6ece5;color:#1f2a44;">Satış</span>
                        </div>
                        <div style="font-family:Poppins,sans-serif;font-weight:600;font-size:20px;">{fmt_int(m['at_risk_vip_count'])} VIP müşteri</div>
                        <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                            Tipik alım aralığını aşan A sınıfı müşteriler; özel iskonto yerine özel temsilci aramasıyla <b>+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</b> korunur.
                        </div>
                    </div>
                    <div style="flex:1 1 250px;min-width:0;border:1px solid #e4dfd6;border-radius:10px;padding:12px 14px;background:#faf7f2;display:flex;flex-direction:column;gap:5px;">
                        <div style="display:flex;gap:6px;flex-wrap:wrap;">
                            <span class="pill" style="background:#d6ece5;color:#1f2a44;">Satış</span>
                            <span class="pill" style="background:#fdf0cf;color:#1f2a44;">Fiyatlama</span>
                        </div>
                        <div style="font-family:Poppins,sans-serif;font-weight:600;font-size:20px;">{fmt_int(m['suspicious_orders_count'])} sipariş satırı</div>
                        <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                            Kategori medyanından 10+ puan fazla indirim verildi; tavan onayıyla <b>+{fmt_tl(m['discount_leak_tl'], 1)}</b> brüt kâr içeride kalır.
                        </div>
                    </div>
                    <div style="flex:1 1 250px;min-width:0;border:1px solid #e4dfd6;border-radius:10px;padding:12px 14px;background:#faf7f2;display:flex;flex-direction:column;gap:5px;">
                        <div style="display:flex;gap:6px;flex-wrap:wrap;">
                            <span class="pill" style="background:#dbe5f8;color:#1f2a44;">Tedarik</span>
                            <span class="pill" style="background:#fbe0d3;color:#1f2a44;">Pazarlama</span>
                        </div>
                        <div style="font-family:Poppins,sans-serif;font-weight:600;font-size:20px;">{fmt_int(m['dead_sku_count'])} hareketsiz ürün</div>
                        <div style="font-size:12px;line-height:1.4;color:#1f2a44;">
                            {m['dead_threshold']}+ gündür satılmayan ürünler, A sınıfı yıldız kitaplarla (%80 ciro) ikili paket yapılarak eritilebilir.
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # BÖLÜM 5: ÖNÜMÜZDEKİ 30 GÜN İÇİN AKSİYON PLANI (İNTERAKTİF SLIDER + SVG ETKİ BARI)
        # ---------------------------------------------------------------------
        render_html(
            """
            <div class="card" style="border-left:4px solid #e0a526;margin-bottom:8px;">
                <h2 class="h"><span class="num">5</span>Önümüzdeki 30 gün için aksiyon planı</h2>
                <p class="sub">Aksiyonları aç/kapat, hedef oranları değiştir · beklenen ciro etkisi aşağıda anında güncellenir</p>
            </div>
            """
        )

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            render_html(
                """
                <div style="display:flex;gap:6px;margin-bottom:4px;">
                    <span class="pill" style="background:#1f2a44;color:#fff;">Pazarlama</span>
                    <span class="pill" style="background:#1f2a44;color:#fff;">Satış</span>
                </div>
                <b style="font-size:13.5px;">1. Geciken VIP müşterileri geri kazan</b>
                """
            )
            r1 = st.slider("Hedef geri kazanım (%)", 10, 60, 35, 5, key="rr_act_r1")
            on1 = st.checkbox("Aktif", value=True, key="rr_act_on1")
            gain1 = (m["vip_recovery_potential_tl"] * (r1 / 35.0)) if on1 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:18px;color:#3d9a84;">+{fmt_tl(gain1, 1)}</div>')

        with ac2:
            render_html(
                """
                <div style="display:flex;gap:6px;margin-bottom:4px;">
                    <span class="pill" style="background:#1f2a44;color:#fff;">Satış</span>
                    <span class="pill" style="background:#1f2a44;color:#fff;">Fiyatlama</span>
                </div>
                <b style="font-size:13.5px;">2. Kategori üstü fazla indirimi sınırla</b>
                """
            )
            r2 = st.slider("Engellenen kaçak oranı (%)", 20, 100, 70, 5, key="rr_act_r2")
            on2 = st.checkbox("Aktif ", value=True, key="rr_act_on2")
            gain2 = (m["discount_leak_tl"] * (r2 / 100.0)) if on2 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:18px;color:#e0a526;">+{fmt_tl(gain2, 1)}</div>')

        with ac3:
            render_html(
                """
                <div style="display:flex;gap:6px;margin-bottom:4px;">
                    <span class="pill" style="background:#1f2a44;color:#fff;">Tedarik</span>
                    <span class="pill" style="background:#1f2a44;color:#fff;">CRM</span>
                </div>
                <b style="font-size:13.5px;">3. A sınıfı stok koruması &amp; 2. sipariş paketi</b>
                """
            )
            r3 = st.slider("Dönüşüm & stok koruma (%)", 10, 60, 30, 5, key="rr_act_r3")
            on3 = st.checkbox("Aktif  ", value=True, key="rr_act_on3")
            base_g3 = m["stockout_prevention_tl"] + m["retention_potential_tl"]
            gain3 = (base_g3 * (r3 / 30.0)) if on3 else 0.0
            render_html(f'<div style="font-family:Poppins,sans-serif;font-weight:600;font-size:18px;color:#3b6fd1;">+{fmt_tl(gain3, 1)}</div>')

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
        end_x = x3 + w3
        band_x1 = max(25.0, end_x - 36.0)
        band_x2 = min(585.0, end_x + 20.0)
        band_mid = (band_x1 + band_x2) / 2.0

        render_html(
            f"""
            <div class="card" style="margin-top:8px;">
                <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;">
                    <svg viewBox="0 0 600 110" width="100%" style="flex:2 1 420px;min-width:0;">
                        <rect x="20" y="46" width="{w_base:.1f}" height="28" fill="#cfc9bd"/>
                        <rect x="{x1:.1f}" y="46" width="{w1:.1f}" height="28" fill="#3d9a84"><title>VIP Kazanım: +{fmt_tl(gain1, 1)}</title></rect>
                        <rect x="{x2:.1f}" y="46" width="{w2:.1f}" height="28" fill="#e0a526"><title>İndirim Disiplini: +{fmt_tl(gain2, 1)}</title></rect>
                        <rect x="{x3:.1f}" y="46" width="{w3:.1f}" height="28" fill="#3b6fd1"><title>Stok &amp; Paket: +{fmt_tl(gain3, 1)}</title></rect>
                        <line x1="{band_x1:.1f}" y1="32" x2="{band_x2:.1f}" y2="32" stroke="#5b6475" stroke-width="2"/>
                        <line x1="{band_x1:.1f}" y1="26" x2="{band_x1:.1f}" y2="38" stroke="#5b6475" stroke-width="2"/>
                        <line x1="{band_x2:.1f}" y1="26" x2="{band_x2:.1f}" y2="38" stroke="#5b6475" stroke-width="2"/>
                        <text x="{band_mid:.1f}" y="20" font-size="10.5" fill="#5b6475" text-anchor="middle">tahmin ±%25 oynayabilir</text>
                        <text x="20" y="94" font-size="11" fill="#1f2a44" font-weight="600">Bir şey yapmazsak {fmt_tl(base_30d)}</text>
                        <text x="580" y="94" font-size="11" fill="#1f2a44" font-weight="600" text-anchor="end">Aksiyonlarla {fmt_tl(tot_30d)}</text>
                    </svg>
                    <div style="flex:1 1 240px;min-width:0;display:flex;flex-direction:column;gap:6px;">
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Üç aksiyondan beklenen 30 günlük ek katkı</div>
                            <div class="v">+{fmt_tl(tot_gain, 1)} <span class="chg pos">+%{gain_pct:.1f}</span></div>
                        </div>
                        <span class="pill" style="background:#d6ece5;color:#14513f;align-self:flex-start;">Stok kapasitesi %86 yeterli</span>
                    </div>
                </div>
            </div>
            """
        )

    # =========================================================================
    # SAYFA 2: VERİ KALİTESİ (rafradar-versiyon2.html 'kalite' Birebir)
    # =========================================================================
    else:
        total_raw_rows = len(scope_raw_df)
        return_rows = int(scope_raw_df["IsReturn"].sum())
        clean_rows = len(scope_sales_df)
        clean_ratio_pct = (clean_rows / max(1, total_raw_rows)) * 100.0

        hdr_left, hdr_right = st.columns([8.2, 1.8])
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
        with hdr_right:
            render_chatbox_widgets(m, sel_lbl, comp_lbl, selected_cat)

        render_html(
            f"""
            <section class="rr-pills">
                <span class="pill pos">{fmt_int(clean_rows)} / {fmt_int(total_raw_rows)} satır kullanıldı · %{clean_ratio_pct:.3f}</span>
                <span class="pill" style="background:#ece7dd;color:#1f2a44;">{sel_lbl}</span>
                <span class="pill" style="background:#ece7dd;color:#1f2a44;">{fmt_int(S['orders'])} sipariş · {fmt_int(S['cust'])} müşteri</span>
            </section>
            """
        )

        # 1. VERİ ÖN İŞLEME · 5 ADIM
        render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">1</span>Veri ön işleme · 5 adım</h2>
                <p class="sub">Ham dosyadan panellere giden yol</p>
                <div class="vq-step-row">
                    <div class="vq-step" style="background:#d6ece5;">
                        <div class="vq-step-label">1 · YÜKLEME</div>
                        <div class="vq-step-val">{fmt_int(total_raw_rows)}</div>
                        <div class="vq-step-desc">satır okundu · 10 sütun</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fdf0cf;">
                        <div class="vq-step-label">2 · TEMİZLİK</div>
                        <div class="vq-step-val">−{fmt_int(return_rows)}</div>
                        <div class="vq-step-desc">sıfır/negatif ve iptal satır çıkarıldı</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#dbe5f8;">
                        <div class="vq-step-label">3 · HESAPLAMA</div>
                        <div class="vq-step-val">+12</div>
                        <div class="vq-step-desc">yeni sütun: brüt ciro, net ciro, indirim</div>
                    </div>
                    <div class="vq-arrow">→</div>
                    <div class="vq-step" style="background:#fbe0d3;">
                        <div class="vq-step-label">4 · SEGMENTLEME</div>
                        <div class="vq-step-val">A · B · C</div>
                        <div class="vq-step-desc">müşteri ve ürün, ciro payına göre</div>
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

        # 2. ALAN DOLULUĞU & 3. HESAPLAMA KURALLARI
        c_q1, c_q2 = st.columns([5.2, 4.8])
        with c_q1:
            fields = [
                ("Sipariş no (InvoiceNo)", 100, "#3d9a84", "%100"),
                ("Müşteri no (CustomerID)", 100, "#3d9a84", "%100"),
                ("Tarih (InvoiceDate)", 100, "#3d9a84", "%100"),
                ("Ürün kodu & adı", 100, "#3d9a84", "%100"),
                ("Kategori (Category)", 100, "#3d9a84", "%100"),
                ("İl (City)", 100, "#3d9a84", "%100"),
                ("Adet, Liste & Net fiyat", 100, "#3d9a84", "%100"),
                ("Alış maliyeti (Liste %44)", 100, "#e0a526", "Varsayım"),
            ]
            rows_html = "".join(
                f'<div style="display:grid;grid-template-columns:165px 1fr 58px;gap:10px;align-items:center;padding:5px 0;">'
                f'<span style="font-size:12.5px;">{fn}</span>'
                f'<svg viewBox="0 0 300 11" width="100%"><rect width="300" height="11" rx="5.5" fill="#efe9de"/><rect width="{w * 3}" height="11" rx="5.5" fill="{col}"/></svg>'
                f'<b style="font-size:11.5px;text-align:right;">{lbl}</b>'
                f"</div>"
                for fn, w, col, lbl in fields
            )
            render_html(
                f"""
                <div class="card">
                    <h2 class="h"><span class="num">2</span>Alan doluluğu</h2>
                    <p class="sub">Her alanın yüzde kaçı dolu · sarı olanlar varsayımla tamamlandı</p>
                    <div style="margin-top:8px;">{rows_html}</div>
                </div>
                """
            )

        with c_q2:
            render_html(
                f"""
                <div class="card">
                    <h2 class="h"><span class="num">3</span>Hesaplama özeti ve kurallar</h2>
                    <p class="sub">Panellerdeki temel formüller</p>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Brüt &amp; Net Ciro</div>
                            <div class="v" style="font-size:13px;">Adet × Liste / Net Fiyat</div>
                        </div>
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Tahmini Brüt Kâr</div>
                            <div class="v" style="font-size:13px;">Net Ciro − (0,44 × Brüt)</div>
                        </div>
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Fazla İndirim Anomalisi</div>
                            <div class="v" style="font-size:13px;">İndirim &gt; Kat. Medyanı +10p</div>
                        </div>
                        <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;">
                            <div class="t">Güvenlik Stoğu (SS)</div>
                            <div class="v" style="font-size:13px;">z × σ_haftalık × √Süre</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;padding:9px 11px;border-radius:8px;background:#fbe3e0;color:#8f2a24;font-size:12px;">
                        <b>Anomali Özeti:</b> Seçili dönemde <b>{fmt_int(m['suspicious_orders_count'])} satırda</b> kategori normu aşımıyla <b>{fmt_tl(m['discount_leak_tl'], 1)}</b> fazla indirim tespit edildi.
                    </div>
                </div>
                """
            )


if __name__ == "__main__":
    render_executive_summary()
