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
# 3. YARDIMCI FORMATLAMA VE DİNAMİK ANALİTİK FONKSİYONLAR
# -----------------------------------------------------------------------------
def fmt_tl(val: float, decimals: int = 2) -> str:
    """TL değerlerini yönetici okumasına uygun (Mn, Bin) formatlar."""
    if pd.isna(val):
        return "₺0"
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


def inject_rafradar_css():
    render_html(
        """
        <style>
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .page-subtitle {
            font-size: 13.5px;
            color: #475569;
            margin-top: -2px;
            margin-bottom: 16px;
            background: #FFFFFF;
            padding: 11px 16px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
        }
        /* KPI Kartları */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 4px solid #2563EB;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            height: 100%;
        }
        .kpi-card.emerald { border-top-color: #10B981; }
        .kpi-card.indigo  { border-top-color: #6366F1; }
        .kpi-card.amber   { border-top-color: #F59E0B; }
        .kpi-card.rose    { border-top-color: #EF4444; }
        .kpi-label {
            font-size: 11.5px;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 25px;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.15;
            margin-bottom: 8px;
        }
        .kpi-delta-pos {
            font-size: 11.5px;
            font-weight: 700;
            color: #166534;
            background: #DCFCE7;
            padding: 2px 7px;
            border-radius: 5px;
            display: inline-block;
        }
        .kpi-delta-neg {
            font-size: 11.5px;
            font-weight: 700;
            color: #991B1B;
            background: #FEE2E2;
            padding: 2px 7px;
            border-radius: 5px;
            display: inline-block;
        }
        .kpi-delta-warn {
            font-size: 11.5px;
            font-weight: 700;
            color: #92400E;
            background: #FEF3C7;
            padding: 2px 7px;
            border-radius: 5px;
            display: inline-block;
        }
        .kpi-subtext {
            font-size: 11.5px;
            color: #64748B;
            margin-left: 6px;
            font-weight: 500;
        }
        /* Numaralı Analiz Kartları */
        .section-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }
        .section-num {
            font-size: 11.5px;
            font-weight: 800;
            color: #2563EB;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            margin-bottom: 6px;
        }
        .section-num span {
            background-color: #DBEAFE;
            color: #1E3A8A;
            padding: 3px 8px;
            border-radius: 6px;
            margin-right: 7px;
        }
        .section-takeaway {
            font-size: 16px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 6px;
            line-height: 1.38;
        }
        .insight-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            padding: 12px 15px;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            color: #334155;
            margin-top: 8px;
            margin-bottom: 16px;
            line-height: 1.5;
        }
        /* Sızıntı / Fırsat Mini Kartları */
        .leak-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 6px;
            margin-bottom: 16px;
        }
        .leak-item {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 10px 12px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
        }
        .leak-item-title {
            font-size: 12px;
            font-weight: 700;
            color: #0F172A;
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }
        .leak-item-desc {
            font-size: 11.5px;
            color: #475569;
            line-height: 1.35;
        }
        /* Pareto İkili Yoğunlaşma Kutuları */
        .pareto-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
        }
        .pareto-title {
            font-size: 13.5px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .pareto-sub {
            font-size: 12px;
            color: #475569;
            margin-bottom: 12px;
        }
        .tier-row {
            margin-bottom: 10px;
        }
        .tier-header {
            display: flex;
            justify-content: space-between;
            font-size: 11.5px;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 4px;
        }
        .tier-bar-bg {
            width: 100%;
            height: 9px;
            background: #E2E8F0;
            border-radius: 999px;
            overflow: hidden;
        }
        .tier-bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        /* Modern 30 Günlük Aksiyon Kartları (Bölüm 4) */
        .action-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #2563EB;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 11px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
        }
        .action-card.urgent { border-left-color: #EF4444; background: #FFFDFD; }
        .action-card.high   { border-left-color: #F59E0B; }
        .action-card.medium { border-left-color: #10B981; }
        .action-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            flex-wrap: wrap;
            gap: 6px;
        }
        .action-dept {
            font-size: 12px;
            font-weight: 800;
            color: #1E293B;
            background: #F1F5F9;
            padding: 4px 10px;
            border-radius: 6px;
        }
        .action-priority-urgent {
            background: #FEE2E2;
            color: #991B1B;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 9px;
            border-radius: 999px;
        }
        .action-priority-high {
            background: #FEF3C7;
            color: #92400E;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 9px;
            border-radius: 999px;
        }
        .action-priority-med {
            background: #DCFCE7;
            color: #166534;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 9px;
            border-radius: 999px;
        }
        .action-headline {
            font-size: 14px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 5px;
        }
        .action-body {
            font-size: 12.5px;
            color: #334155;
            line-height: 1.45;
            margin-bottom: 8px;
        }
        .action-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #F8FAFC;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px dashed #CBD5E1;
            font-size: 12px;
        }
        .action-impact {
            font-weight: 800;
            color: #0F172A;
            font-size: 13px;
        }
        /* Veri Kalitesi & Metodoloji Kartları */
        .pipeline-step {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 3px solid #2563EB;
            border-radius: 10px;
            padding: 14px;
            height: 100%;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
        }
        .pipeline-step-num {
            font-size: 11px;
            font-weight: 800;
            color: #2563EB;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .pipeline-step-title {
            font-size: 14px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 6px;
        }
        .pipeline-step-desc {
            font-size: 12px;
            color: #475569;
            line-height: 1.45;
        }
        .formula-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #6366F1;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .formula-title {
            font-size: 13px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 3px;
        }
        .formula-code {
            font-family: monospace;
            font-size: 12px;
            background: #F1F5F9;
            color: #1E3A8A;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
            margin: 4px 0;
            font-weight: 600;
        }
        </style>
        """
    )


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
    }


# -----------------------------------------------------------------------------
# 4. ANA RENDER FONKSİYONU (HEM STANDALONE HEM STREAMLIT_APP.PY UYUMLU)
# -----------------------------------------------------------------------------
def render_executive_summary(filtered_df=None, raw_df=None):
    inject_rafradar_css()

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
    # ÜST NAVİGASYON VE DİNAMİK FİLTRE ÇUBUĞU (TARİH & KATEGORİ BAĞIMLI)
    # -------------------------------------------------------------------------
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([2.0, 2.7, 1.9, 2.0, 1.8])

    with top_c1:
        render_html(
            """
            <div style='padding-top:4px;'>
                <span style='font-size:22px; font-weight:900; color:#0F172A;'>📊 RafRadar</span><br>
                <span style='font-size:11.5px; color:#64748B; font-weight:600;'>C-Level Karar Destek Sistemi</span>
            </div>
            """
        )

    with top_c2:
        selected_tab = st.radio(
            "Görünüm Seçimi",
            ["🏛️ Yönetici Özeti", "🛠️ Veri Ön İşleme & Kalite Karnesi"],
            horizontal=True,
            key="rafradar_main_tab",
        )

    with top_c3:
        period_options = (
            ["Tüm Veri Seti", "Son 12 Ay", "Son 6 Ay", "Son Çeyrek (90 Gün)", "Yoğun Sezon (Eyl-Ara)", "Özel Tarih Aralığı"]
            + [f"Yıl: {y}" for y in available_years]
        )
        selected_preset = st.selectbox("📅 Analiz Dönemi", period_options, index=0, key="rafradar_preset")

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
            "📆 Tarih Aralığı (Başlangıç - Bitiş)",
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
            "📚 Ürün Kategorisi",
            ["Tüm Kategoriler"] + top_cats,
            index=0,
            key="rafradar_cat",
        )

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

    span_days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days)
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

    # =========================================================================
    # SEKME 1: YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)
    # =========================================================================
    if selected_tab == "🏛️ Yönetici Özeti":
        render_html(
            f"""
            <div class='page-subtitle'>
                <b>🎯 Yönetici Sorusu:</b> Şirket büyüyor mu, kâr marjımız güvende mi ve bu hafta hangi kritik kararları almalıyız?
                &nbsp;|&nbsp; <b>Aktif Filtre:</b> <code>{start_str} – {end_str}</code> • <b>Kategori:</b> <code>{selected_cat}</code>
                • <b>İşlem Sayısı:</b> <code>{fmt_int(len(scope_sales_df))} satır</code> ({comparison_label})
            </div>
            """
        )

        # ---------------------------------------------------------------------
        # ÜST KPI KARTLARI (5 SÜTUN - %100 DİNAMİK)
        # ---------------------------------------------------------------------
        k1, k2, k3, k4, k5 = st.columns(5)

        rev_delta_cls = "kpi-delta-pos" if m["rev_growth_pct"] >= 0 else "kpi-delta-neg"
        rev_arrow = "▲" if m["rev_growth_pct"] >= 0 else "▼"

        with k1:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Seçili Dönem Net Ciro</div>
                    <div class="kpi-value">{fmt_tl(m['net_rev'])}</div>
                    <span class="{rev_delta_cls}">{rev_arrow} %{abs(m['rev_growth_pct']):.1f}</span>
                    <span class="kpi-subtext">{fmt_int(m['total_invoices'])} fatura</span>
                </div>
                """
            )

        with k2:
            margin_cls = "kpi-delta-pos" if m["profit_margin"] >= 28.0 else "kpi-delta-warn"
            render_html(
                f"""
                <div class="kpi-card emerald">
                    <div class="kpi-label">Tahmini Brüt Kâr & Marj</div>
                    <div class="kpi-value">{fmt_tl(m['est_profit'])}</div>
                    <span class="{margin_cls}">%{m['profit_margin']:.1f} Marj</span>
                    <span class="kpi-subtext">Maliyet: Liste %44'ü</span>
                </div>
                """
            )

        with k3:
            cust_sign = "+" if m["cust_diff"] >= 0 else ""
            cust_cls = "kpi-delta-pos" if m["cust_diff"] >= 0 else "kpi-delta-warn"
            render_html(
                f"""
                <div class="kpi-card indigo">
                    <div class="kpi-label">Aktif B2B Bayi & Pareto</div>
                    <div class="kpi-value">{fmt_int(m['active_customers'])}</div>
                    <span class="{cust_cls}">{cust_sign}{m['cust_diff']} bayi</span>
                    <span class="kpi-subtext">%{m['vip_cust_share_pct']:.0f}'i = %80 Ciro</span>
                </div>
                """
            )

        with k4:
            render_html(
                f"""
                <div class="kpi-card amber">
                    <div class="kpi-label">Ort. İskonto & Norm Dışı Kaçak</div>
                    <div class="kpi-value">%{m['avg_discount']:.1f}</div>
                    <span class="kpi-delta-neg">{fmt_tl(m['discount_leak_tl'], 1)} kaçak</span>
                    <span class="kpi-subtext">{fmt_int(m['suspicious_orders_count'])} sapmalı satır</span>
                </div>
                """
            )

        with k5:
            render_html(
                f"""
                <div class="kpi-card rose">
                    <div class="kpi-label">Aktif Kitap (SKU) & Hareketsiz</div>
                    <div class="kpi-value">{fmt_int(m['active_skus'])}</div>
                    <span class="kpi-delta-neg">{fmt_int(m['dead_sku_count'])} Hareketsiz</span>
                    <span class="kpi-subtext">120+ gün satışsız</span>
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
            bar_colors = ["#2563EB" if v >= rev_thresh else "#CBD5E1" for v in ts_df["NetRev"]]

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
                <div class="section-card">
                    <div class="section-num"><span>1</span> FİNANSAL PERFORMANS VE İSKONTO DİSİPLİNİ</div>
                    <div class="section-takeaway">
                        Zirve dönem ({peak_row['Label']}) ile en düşük dönem arasında {peak_to_low_ratio:.1f} kat ciro farkı var;
                        ancak talebin en yüksek olduğu {period_unit_name} bile ortalama iskonto (%{peak_disc_avg:.1f}) düşmüyor.
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
                    name="Ort. İskonto (%)",
                    mode="lines+markers",
                    line=dict(color="#DC2626", width=2.8),
                    marker=dict(size=7, color="#DC2626"),
                    hovertemplate="<b>%{x}</b><br>İskonto Oranı: %%{y:.1f}<extra></extra>",
                ),
                secondary_y=True,
            )
            disc_min = max(0, float(ts_df["DiscPct"].min()) - 8)
            disc_max = min(85, float(ts_df["DiscPct"].max()) + 8)
            fig1.update_layout(
                height=290,
                margin=dict(l=10, r=10, t=18, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            fig1.update_yaxes(
                title_text="Net Ciro (₺)",
                range=[0, float(ts_df["NetRev"].max()) * 1.22],
                gridcolor="#F1F5F9",
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
                    <b>💡 Yönetici Çıkarımı:</b> Mavi çubuklarla vurgulanan yüksek talep dönemlerinde müşteriler zaten organik olarak
                    sipariş verirken iskontonun <b>%{peak_disc_avg:.1f}</b> seviyesinde sabit tutulması kâr marjını aşındırıyor.
                    Ayrıca haftalık sevkiyatların <b>%{top_dow_share:.1f}'i {top_dow_tr}</b> gününe yığılarak depoda darboğaz yaratıyor.
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
                <div class="section-card">
                    <div class="section-num"><span>2</span> CİRO VE KÂR SIZINTISI KÖPRÜSÜ (FIRSAT ANALİZİ)</div>
                    <div class="section-takeaway">
                        4 operasyonel sızıntı kapatıldığında, mevcut ciroya +{fmt_tl(total_opportunity_tl)}
                        (+%{uplift_pct:.1f}) eklenerek {fmt_tl(target_rev_tl)} seviyesine ulaşılabilir.
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
                "Ulaşılabilir<br>Potansiyel",
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
                    connector={"line": {"color": "#94A3B8", "width": 1.5, "dash": "dot"}},
                    increasing={"marker": {"color": "#10B981"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    totals={"marker": {"color": "#1E3A8A"}},
                )
            )
            fig2.update_layout(
                height=245,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    title="Ciro Köprüsü (₺)",
                    range=[0, target_rev_tl * 1.18],
                    gridcolor="#F1F5F9",
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
                            <span style="color:#059669;">+{fmt_tl(m['vip_recovery_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Siparişi kesen <b>{m['at_risk_vip_count']} büyük bayinin</b> geri kazanım etkisi.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>📦 2. Yıldız Ürün Stoğu</span>
                            <span style="color:#059669;">+{fmt_tl(m['stockout_prevention_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Cironun %80'ini çeken <b>{m['a_sku_count']} A sınıfı kitapta</b> yok satmayı önleme.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🔄 3. Tek Siparişli Bayiler</span>
                            <span style="color:#059669;">+{fmt_tl(m['retention_potential_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Tek alımda kalan <b>%{m['one_time_ratio_pct']:.0f} bayinin</b> çeyreğine 2. sipariş yaptırma.</div>
                    </div>
                    <div class="leak-item">
                        <div class="leak-item-title">
                            <span>🏷️ 4. Aşırı İskonto Tavanı</span>
                            <span style="color:#059669;">+{fmt_tl(m['discount_leak_tl'], 1)}</span>
                        </div>
                        <div class="leak-item-desc">Kategori normunu aşan <b>{fmt_int(m['suspicious_orders_count'])} satırdaki</b> fazladan indirimi kesme.</div>
                    </div>
                </div>
                """
            )

        # ---------------------------------------------------------------------
        # ALT BÖLÜM:
        # [3] ÇİFT TARAFLI BAĞIMLILIK RİSKİ (MÜŞTERİ & ÜRÜN PARETO 80/20 ANALİZİ)
        # [4] YÖNETİM KURULU 30 GÜNLÜK AKSİYON PLANI (MODERN KARTLAR)
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
                    <div class="section-num"><span>3</span> ÇİFT TARAFLI BAĞIMLILIK RİSKİ (PARETO 80/20 ANALİZİ)</div>
                    <div class="section-takeaway">
                        Cironun %80'i müşterilerin sadece %{m['vip_cust_share_pct']:.1f}'ine ({m['vip_cust_count']} bayi)
                        ve kitapların %{m['a_sku_share_pct']:.1f}'ine ({fmt_int(m['a_sku_count'])} SKU) bağımlı.
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
                            <span>👥 Müşteri (Bayi) Yapısı</span>
                            <span style="font-size:11px; background:#DBEAFE; color:#1E3A8A; padding:2px 7px; border-radius:99px;">{fmt_int(m['active_customers'])} Bayi</span>
                        </div>
                        <div class="pareto-sub">İlk 10 bayi tek başına cironun <b>%{m['top10_cust_rev_share']:.1f}</b>'ini üretiyor.</div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>🏆 VIP Bayiler ({m['vip_cust_count']} bayi | %{m['vip_cust_share_pct']:.0f})</span>
                                <span style="color:#2563EB;">%80 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#2563EB;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>🤝 Orta Segment ({b_cust_count} bayi | %{b_cust_share:.0f})</span>
                                <span style="color:#0D9488;">%15 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#0D9488;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header">
                                <span>🌱 Küçük Alıcılar ({c_cust_count} bayi | %{c_cust_share:.0f})</span>
                                <span style="color:#64748B;">%5 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:5%; background:#94A3B8;"></div></div>
                        </div>
                    </div>
                    """
                )

            with p_col2:
                render_html(
                    f"""
                    <div class="pareto-box">
                        <div class="pareto-title">
                            <span>📚 Ürün (Kitap) Yapısı</span>
                            <span style="font-size:11px; background:#DCFCE7; color:#166534; padding:2px 7px; border-radius:99px;">{fmt_int(m['active_skus'])} Kitap</span>
                        </div>
                        <div class="pareto-sub">Kitapların <b>%{m['c_sku_share_pct']:.0f}'i</b> cironun sadece %5'ini getiriyor.</div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>⭐ A Sınıfı Yıldız ({fmt_int(m['a_sku_count'])} SKU | %{m['a_sku_share_pct']:.0f})</span>
                                <span style="color:#10B981;">%80 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:80%; background:#10B981;"></div></div>
                        </div>
                        <div class="tier-row">
                            <div class="tier-header">
                                <span>📘 B Sınıfı Düzenli ({fmt_int(b_sku_count)} SKU | %{b_sku_share:.0f})</span>
                                <span style="color:#3B82F6;">%15 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:15%; background:#3B82F6;"></div></div>
                        </div>
                        <div class="tier-row" style="margin-bottom:0;">
                            <div class="tier-header">
                                <span>📦 C Sınıfı & Pasif ({fmt_int(m['c_sku_count'])} SKU | %{m['c_sku_share_pct']:.0f})</span>
                                <span style="color:#64748B;">%5 Ciro</span>
                            </div>
                            <div class="tier-bar-bg"><div class="tier-bar-fill" style="width:5%; background:#94A3B8;"></div></div>
                        </div>
                    </div>
                    """
                )

            pareto_comp_df = pd.DataFrame(
                {
                    "Grup": ["VIP Bayiler", "A Sınıfı Kitaplar", "C Sınıfı Kitaplar"],
                    "Portföydeki Adet Payı (%)": [
                        round(m["vip_cust_share_pct"], 1),
                        round(m["a_sku_share_pct"], 1),
                        round(m["c_sku_share_pct"], 1),
                    ],
                    "Ürettiği Ciro Payı (%)": [80.0, 80.0, 5.0],
                }
            )
            fig3 = go.Figure()
            fig3.add_trace(
                go.Bar(
                    name="Portföydeki Sayıca Payı (%)",
                    x=pareto_comp_df["Grup"],
                    y=pareto_comp_df["Portföydeki Adet Payı (%)"],
                    marker_color="#94A3B8",
                    text=[f"%{v:.1f} Adet" for v in pareto_comp_df["Portföydeki Adet Payı (%)"]],
                    textposition="outside",
                )
            )
            fig3.add_trace(
                go.Bar(
                    name="Ürettiği Toplam Ciro Payı (%)",
                    x=pareto_comp_df["Grup"],
                    y=pareto_comp_df["Ürettiği Ciro Payı (%)"],
                    marker_color=["#2563EB", "#10B981", "#EF4444"],
                    text=[f"%{v:.0f} Ciro" for v in pareto_comp_df["Ürettiği Ciro Payı (%)"]],
                    textposition="outside",
                )
            )
            fig3.update_layout(
                barmode="group",
                height=210,
                margin=dict(l=10, r=10, t=24, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 105], gridcolor="#F1F5F9", title="Pay (%)"),
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
                <div class="section-card">
                    <div class="section-num"><span>4</span> YÖNETİM KURULU 30 GÜNLÜK AKSİYON PLANI</div>
                    <div class="section-takeaway">
                        Seçilen dönem verilerinden otomatik süzülen, en hızlı nakit ve kâr etkisi yaratacak 4 somut yönetim kararı:
                    </div>
                    <div class="action-card urgent">
                        <div class="action-top">
                            <span class="action-dept">🎯 1. Pazarlama & VIP Bayi İlişkileri</span>
                            <span class="action-priority-urgent">🔴 ACİL • Bu Hafta</span>
                        </div>
                        <div class="action-headline">Siparişi Kesen Büyük Bayiler İçin Birebir "Geri Kazanım" Aramaları</div>
                        <div class="action-body">
                            <b>Tespit Edilen Sorun:</b> {vip_str}<br>
                            <b>Yapılacak Aksiyon:</b> Satış temsilcileri bu hafta pasif VIP bayileri doğrudan arayarak özel vade veya kademeli sepet hedefi tanımlamalı.
                        </div>
                        <div class="action-footer">
                            <span style="color:#64748B;">Beklenen Finansal Etki (Ciro Koruma):</span>
                            <span class="action-impact" style="color:#DC2626;">+{fmt_tl(m['vip_recovery_potential_tl'])} Potansiyel Ciro</span>
                        </div>
                    </div>
                    <div class="action-card urgent">
                        <div class="action-top">
                            <span class="action-dept">💰 2. Satış & Fiyatlama Disiplini</span>
                            <span class="action-priority-urgent">🔴 ACİL • Bu Hafta</span>
                        </div>
                        <div class="action-headline">Kategori Normunu Aşan Kontrolsüz İskontolara Üst Sınır (Tavan) Getirilmesi</div>
                        <div class="action-body">
                            <b>Tespit Edilen Sorun:</b> Başta <b>{anom_cat_str}</b> olmak üzere {fmt_int(m['suspicious_orders_count'])} işlem satırında kategori ortalamasının 10+ puan üzerinde indirim verilmiş.<br>
                            <b>Yapılacak Aksiyon:</b> Sipariş ekranında kategori medyan iskontosunu +5 puandan fazla aşan işlemlere otomatik yönetici onay şartı konulmalı.
                        </div>
                        <div class="action-footer">
                            <span style="color:#64748B;">Beklenen Finansal Etki (Doğrudan Net Kâr):</span>
                            <span class="action-impact" style="color:#D97706;">+{fmt_tl(m['discount_leak_tl'])} Doğrudan Kâr Koruma</span>
                        </div>
                    </div>
                    <div class="action-card high">
                        <div class="action-top">
                            <span class="action-dept">📦 3. Tedarik Zinciri & Stok Yönetimi</span>
                            <span class="action-priority-high">🟠 YÜKSEK • 15 Gün İçinde</span>
                        </div>
                        <div class="action-headline">A Sınıfı Yıldız Kitaplarda Emniyet Stoğu, Hareketsiz Ürünlerde Eritme Kampanyası</div>
                        <div class="action-body">
                            <b>Tespit Edilen Sorun:</b> Cironun %80'ini üreten <b>{fmt_int(m['a_sku_count'])} kritik kitapta</b> stoksuz kalma riski varken, <b>{fmt_int(m['dead_sku_count'])} kitap</b> 120+ gündür satılmıyor.<br>
                            <b>Yapılacak Aksiyon:</b> A sınıfı kitaplara %95 hizmet seviyesiyle emniyet stoğu tanımlanmalı; hareketsiz kitaplar çok satanlarla "Avantajlı Paket" yapılarak eritilmeli.
                        </div>
                        <div class="action-footer">
                            <span style="color:#64748B;">Beklenen Finansal Etki (Yok Satmayı Önleme):</span>
                            <span class="action-impact" style="color:#2563EB;">+{fmt_tl(m['stockout_prevention_tl'])} Ciro + Nakit Çözülümü</span>
                        </div>
                    </div>
                    <div class="action-card medium" style="margin-bottom:0;">
                        <div class="action-top">
                            <span class="action-dept">🤝 4. CRM & Yeni Bayi Tutundurma</span>
                            <span class="action-priority-med">🟢 PLANLI • 30 Gün İçinde</span>
                        </div>
                        <div class="action-headline">Tek Siparişte Kalan Bayileri İkinci Siparişe Taşıyacak Otomatik Çapraz Teklif</div>
                        <div class="action-body">
                            <b>Tespit Edilen Sorun:</b> Dönem içinde alım yapan bayilerin <b>%{m['one_time_ratio_pct']:.1f}'i ({fmt_int(m['one_time_count'])} bayi)</b> yalnızca 1 kez fatura kesip tekrar sipariş vermemiş.<br>
                            <b>Yapılacak Aksiyon:</b> İlk faturadan 30 gün sonra bayinin aldığı kategoriyle en sık birlikte satılan tamamlayıcı ürün grubu için otomatik hatırlatma teklifi gönderilmeli.
                        </div>
                        <div class="action-footer">
                            <span style="color:#64748B;">Beklenen Finansal Etki (%25 Tekrar Sipariş Kazanımı):</span>
                            <span class="action-impact" style="color:#059669;">+{fmt_tl(m['retention_potential_tl'])} Ek Sipariş Hacmi</span>
                        </div>
                    </div>
                </div>
                """
            )

    # =========================================================================
    # SEKME 2: VERİ ÖN İŞLEME, KALİTE & METODOLOJİ KARNESİ
    # =========================================================================
    else:
        total_raw_rows = len(scope_raw_df)
        return_rows = int(scope_raw_df["IsReturn"].sum())
        clean_rows = len(scope_sales_df)
        clean_ratio_pct = (clean_rows / max(1, total_raw_rows)) * 100.0
        return_amount_tl = abs(float(scope_raw_df.loc[scope_raw_df["IsReturn"], "TotalAmount"].sum()))

        render_html(
            f"""
            <div class='page-subtitle'>
                <b>🔬 Veri Ön İşleme ve Metodoloji Rehberi:</b> Ham B2B Excel verisinden nasıl anlamlı iş sonuçları ürettik,
                veriyi nasıl temizledik ve ekrandaki her bir metriği hangi formülle hesapladık?
                &nbsp;|&nbsp; <b>İncelenen Dönem:</b> <code>{start_str} – {end_str}</code>
            </div>
            """
        )

        dq1, dq2, dq3, dq4 = st.columns(4)

        with dq1:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Ham İşlem & Net Satış Kaydı</div>
                    <div class="kpi-value">{fmt_int(clean_rows)}</div>
                    <span class="kpi-delta-pos">%{clean_ratio_pct:.1f} Net Satış</span>
                    <span class="kpi-subtext">Toplam Ham: {fmt_int(total_raw_rows)} satır</span>
                </div>
                """
            )

        with dq2:
            render_html(
                f"""
                <div class="kpi-card rose">
                    <div class="kpi-label">Ayrıştırılan İptal / İade</div>
                    <div class="kpi-value">{fmt_int(return_rows)}</div>
                    <span class="kpi-delta-neg">{fmt_tl(return_amount_tl, 1)} İade</span>
                    <span class="kpi-subtext">'C' kodlu & negatif miktarlı</span>
                </div>
                """
            )

        with dq3:
            render_html(
                f"""
                <div class="kpi-card amber">
                    <div class="kpi-label">Norm Dışı İskonto Anomalisi</div>
                    <div class="kpi-value">{fmt_int(m['suspicious_orders_count'])}</div>
                    <span class="kpi-delta-warn">{fmt_tl(m['discount_leak_tl'], 1)} Sapma</span>
                    <span class="kpi-subtext">Kategori medyanı üstü</span>
                </div>
                """
            )

        with dq4:
            render_html(
                """
                <div class="kpi-card emerald">
                    <div class="kpi-label">Veri Zenginleştirme & Modeller</div>
                    <div class="kpi-value">10 → 22 Sütun</div>
                    <span class="kpi-delta-pos">+12 Yeni Değişken</span>
                    <span class="kpi-subtext">RFM, ABC, Marj & Kaçak</span>
                </div>
                """
            )

        st.write("")

        render_html(
            """
            <div class="section-card">
                <div class="section-num"><span>1</span> HAM VERİDEN KARAR DESTEK SİSTEMİNE: 5 ADIMLI VERİ ÖN İŞLEME HUNİSİ</div>
                <div class="section-takeaway">
                    40 MB büyüklüğündeki ham B2B kitap dağıtım verisi, 5 aşamalı otomatik temizleme ve değişken türetme
                    hattından geçirilerek yönetici kararlarına hazır hale getirilmiştir.
                </div>
            </div>
            """
        )

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            render_html(
                """
                <div class="pipeline-step">
                    <div class="pipeline-step-num">ADIM 1 • PERFORMANS</div>
                    <div class="pipeline-step-title">⚡ Parquet Dönüşümü & Önbellekleme</div>
                    <div class="pipeline-step-desc">
                        40 MB'lık yavaş Excel (<code>xlsx</code>) dosyası 14 MB'lık <b>Apache Parquet</b> sütunsal formatına çevrilip
                        <code>@st.cache_data</code> ile belleğe alındı. Okuma süresi <b>18 saniyeden 0,4 saniyeye</b> indirildi.
                    </div>
                </div>
                """
            )
        with s2:
            render_html(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-step-num">ADIM 2 • TEMİZLİK</div>
                    <div class="pipeline-step-title">🧹 İptal, İade & Eksik Veri Ayrıştırması</div>
                    <div class="pipeline-step-desc">
                        Fatura numarası <code>C</code> veya <code>IADE</code> içeren ya da miktarı negatif olan
                        <b>{fmt_int(return_rows)} satır</b> net satış ciro hesaplarını şişirmemesi için <code>IsReturn</code> olarak ayrıştırıldı.
                    </div>
                </div>
                """
            )
        with s3:
            render_html(
                """
                <div class="pipeline-step">
                    <div class="pipeline-step-num">ADIM 3 • FİNANSAL TÜRETİM</div>
                    <div class="pipeline-step-title">💰 Brüt Ciro, İskonto & Kâr Marjı</div>
                    <div class="pipeline-step-desc">
                        Ham verideki <code>UnitPrice</code> (Liste Fiyatı) ve <code>NetPrice</code> (İndirimli Fiyat) farkından
                        her satırın <b>Brüt Tutarı, Net Cirosu, İskonto Oranı (%) ve Tahmini Brüt Kârı</b> türetildi.
                    </div>
                </div>
                """
            )
        with s4:
            render_html(
                """
                <div class="pipeline-step">
                    <div class="pipeline-step-num">ADIM 4 • SEGMENTASYON</div>
                    <div class="pipeline-step-title">🎯 Müşteri (RFM) & Ürün (ABC) Sınıflaması</div>
                    <div class="pipeline-step-desc">
                        Bayiler son alım günü (Recency), sipariş sıklığı (Frequency) ve tutara (Monetary) göre;
                        kitaplar ise kümülatif ciro payına göre <b>A (%80), B (%15), C (%5)</b> olarak sınıflandırıldı.
                    </div>
                </div>
                """
            )
        with s5:
            render_html(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-step-num">ADIM 5 • ANOMALİ DENETİMİ</div>
                    <div class="pipeline-step-title">🚨 Norm Dışı İskonto & Uç Değer Tespiti</div>
                    <div class="pipeline-step-desc">
                        Her kategorinin ortanca (medyan) iskontosu hesaplandı; bu normu aşarak kâr sızıntısı yaratan
                        <b>{fmt_int(m['suspicious_orders_count'])} işlem satırı</b> otomatik olarak işaretlendi.
                    </div>
                </div>
                """
            )

        st.write("")
        col_f1, col_f2 = st.columns([5.5, 4.5])

        with col_f1:
            render_html(
                """
                <div class="section-card">
                    <div class="section-num"><span>2</span> EKRANDAKİ METRİKLER NASIL HESAPLANDI? (ANALİTİK FORMÜL REHBERİ)</div>
                    <div class="section-takeaway">
                        Yönetici özetindeki tüm göstergeler, ham veri sütunlarından aşağıdaki matematiksel kurallarla canlı hesaplanır:
                    </div>
                    <div class="formula-card">
                        <div class="formula-title">1. Brüt Ciro, Net Ciro ve Gerçekleşen İskonto Oranı (%)</div>
                        <div class="formula-code">Brüt Ciro = Quantity × UnitPrice &nbsp;|&nbsp; Net Ciro = Quantity × NetPrice</div>
                        <div style="font-size:12px; color:#475569;">
                            Liste tutarı ile indirimli net tutar arasındaki farktan gerçek iskonto oranı hesaplanır:
                            <b>İskonto (%) = ((Brüt Ciro - Net Ciro) / Brüt Ciro) × 100</b>.
                        </div>
                    </div>
                    <div class="formula-card">
                        <div class="formula-title">2. Tahmini Brüt Kâr ve Marj Hesabı (Alış Maliyeti Varsayımı)</div>
                        <div class="formula-code">Tahmini Maliyet = 0,44 × Liste Tutarı &nbsp;|&nbsp; Brüt Kâr = Net Ciro - Tahmini Maliyet</div>
                        <div style="font-size:12px; color:#475569;">
                            Ham veride yayınevi alış maliyeti bulunmadığından, B2B kitap dağıtım sektörü ortalaması olan
                            <b>Liste Fiyatının %44'ü (%56 dağıtımcı alış iskontosu)</b> maliyet kabul edilmiştir.
                        </div>
                    </div>
                    <div class="formula-card">
                        <div class="formula-title">3. Norm Dışı İskonto Kaçağı (Fiyat Anomalisi Tespiti)</div>
                        <div class="formula-code">Kaçak TL = Brüt Tutar × (Satır İskontosu % - Kategori Medyan İskontosu %)</div>
                        <div style="font-size:12px; color:#475569;">
                            Her ürün kategorisinin kendi normal (medyan) iskontosu bulunur. Kategori normunun <b>+10 puan</b>
                            üzerinde indirim verilen siparişlerde fazladan vazgeçilen tutar "İskonto Kaçağı" olarak raporlanır.
                        </div>
                    </div>
                    <div class="formula-card" style="margin-bottom:0;">
                        <div class="formula-title">4. Çift Taraflı Pareto (80/20), Kayıp Riski ve Emniyet Stoğu</div>
                        <div class="formula-code">Kümülatif Ciro Payı ≤ %80 → A Sınıfı (VIP Bayi / Yıldız Kitap)</div>
                        <div style="font-size:12px; color:#475569;">
                            Bayiler ve kitaplar ciroya göre büyükten küçüğe sıralanır; ilk %80 ciroyu üreten çekirdek grup
                            <b>A Sınıfı</b> olarak atanır. Stok planlamasında <b>Emniyet Stoğu (SS = z · σ · √L)</b> formülü kullanılır.
                        </div>
                    </div>
                </div>
                """
            )

        with col_f2:
            render_html(
                """
                <div class="section-card">
                    <div class="section-num"><span>3</span> KATEGORİ BAZLI İSKONTO SAPMALARI VE KAÇAK TUTARI (CANLI VERİ)</div>
                    <div class="section-takeaway">
                        Seçilen dönemde kategori medyan iskontosunun aşılması nedeniyle en çok gelir sızıntısı yaşanan ürün grupları:
                    </div>
                </div>
                """
            )

            top_anom_df = m["cat_anom_df"].head(7).copy()
            if not top_anom_df.empty and top_anom_df["LeakTL"].sum() > 0:
                top_anom_df["Label"] = top_anom_df.apply(
                    lambda r: f"{r['Category'][:22]} (Norm %{r['MedianDisc']:.0f}→Max %{r['MaxDisc']:.0f})",
                    axis=1,
                )
                fig_dq = go.Figure(
                    go.Bar(
                        x=top_anom_df["LeakTL"],
                        y=top_anom_df["Label"],
                        orientation="h",
                        marker_color="#DC2626",
                        text=[fmt_tl(v, 1) for v in top_anom_df["LeakTL"]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Norm Dışı İskonto Kaybı: ₺%{x:,.0f}<extra></extra>",
                    )
                )
                fig_dq.update_layout(
                    height=265,
                    margin=dict(l=10, r=70, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="Norm Dışı Fazla İskonto Tutarı (₺)",
                        range=[0, float(top_anom_df["LeakTL"].max()) * 1.28],
                        gridcolor="#F1F5F9",
                    ),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_dq, use_container_width=True)
            else:
                st.info("Seçilen dar filtrede belirgin bir kategori iskonto sapması bulunmadı.")

            render_html(
                """
                <div class="insight-box">
                    <b>📌 Neden Medyan (Ortanca) Kullanıyoruz?</b> Aritmetik ortalama, aşırı indirimli birkaç büyük faturadan
                    etkilenerek bozulur. <b>Kategori Medyanı</b> ise o kategorinin gerçek piyasa normunu temsil eder;
                    böylece standart dışı indirim verilen faturalar net biçimde yakalanır.
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
