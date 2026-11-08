import os
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. Sayfa ve Genel Ayarlar (RafRadar v2 Uyumlu)
# ==========================================
st.set_page_config(
    page_title="RafRadar · Yönetici Özeti",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_yonetici_ozeti_tab(filtered_df=None, raw_df=None):
    """
    'yönetici özeti.py' dosyasını güvenli şekilde çalıştırır.
    Dosya içinde st.set_page_config() çağrılsa bile çakışmayı önler ve
    RafRadar v2 sol menü + üst KPI yapısını çizer.
    """
    exec_path = os.path.join(os.path.dirname(__file__), "yönetici özeti.py")
    if not os.path.exists(exec_path):
        st.error("'yönetici özeti.py' dosyası bulunamadı.")
        return

    orig_set_page_config = st.set_page_config
    try:
        st.set_page_config = lambda *args, **kwargs: None
        spec = importlib.util.spec_from_file_location("yonetici_ozeti", exec_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "render_executive_summary"):
            module.render_executive_summary(filtered_df, raw_df)
    finally:
        st.set_page_config = orig_set_page_config


# ==========================================
# 2. Veri Yükleme, Ön İşleme ve Önbellekleme
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Veri setini okur, temizler ve analize hazır türetilmiş değişkenler ekler.
    Performans optimizasyonu için Parquet formatını önceliklendirir.
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
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate']).copy()

    df['InvoiceNo_Str'] = df['InvoiceNo'].astype(str).str.strip()
    df['IsReturn'] = (
        df['InvoiceNo_Str'].str.upper().str.startswith('C')
        | df['InvoiceNo_Str'].str.upper().str.contains('IADE')
        | (df['Quantity'] < 0)
    )

    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)

    if 'NetPrice' in df.columns:
        df['NetPrice'] = pd.to_numeric(df['NetPrice'], errors='coerce').fillna(df['UnitPrice'])
    else:
        df['NetPrice'] = df['UnitPrice']

    df['GrossAmount'] = df['Quantity'] * df['UnitPrice']
    df['TotalAmount'] = df['Quantity'] * df['NetPrice']
    df['DiscountAmount'] = np.maximum(0.0, df['GrossAmount'] - df['TotalAmount'])

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Date'] = df['InvoiceDate'].dt.date

    if 'City' in df.columns:
        df['City'] = df['City'].fillna('Bilinmiyor').astype(str).str.strip()
    else:
        df['City'] = 'Bilinmiyor'

    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('Genel / Diğer').astype(str).str.strip()
    else:
        df['Category'] = 'Genel Kitap'

    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('Tanımsız Kitap').astype(str).str.strip()
    else:
        df['Description'] = 'Tanımsız Kitap'

    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].fillna('Anonim Bayi').astype(str).str.strip()
    else:
        df['CustomerID'] = df['City']

    return df


# ==========================================
# 3. Ana Görünüm (RafRadar v2 Ortak Sol Menü & Üst Bar)
# ==========================================
render_yonetici_ozeti_tab()
