import streamlit as st
import pandas as pd
import plotly.express as px

# Sayfa Konfigürasyonu
st.set_page_config(page_title="B2B Kitap Dağıtım Dashboard", layout="wide")

st.title("📚 B2B Kitap Dağıtım Karar Destek Sistemi")

# Veri Yükleme (Cache kullanarak performansı artırıyoruz)
@st.cache_data
def load_data():
    df = pd.read_excel("B2B_Transaction_Data_.xlsx")
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_data()

# Kenar Çubuğu (Sidebar) Filtreleri
st.sidebar.header("Filtreler")
selected_city = st.sidebar.multiselect("Şehir Seçin", options=df["City"].unique(), default=df["City"].unique())
filtered_df = df[df["City"].isin(selected_city)]

# Sekme Yapısı (Satış, Pazarlama, Tedarik Zinciri)
tab_sales, tab_mkt, tab_scm = st.tabs(["📊 Satış Yönetimi", "🎯 Pazarlama", "🚚 Tedarik Zinciri"])

with tab_sales:
    st.header("Satış Performansı")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Ciro", f"₺{filtered_df['TotalAmount'].sum():,.2f}")
    col2.metric("Toplam Satış Adedi", f"{filtered_df['Quantity'].sum():,}")
    col3.metric("Sipariş Sayısı", f"{filtered_df['InvoiceNo'].nunique():,}")
    
    # Zamana Göre Satış Grafiği
    daily_sales = filtered_df.groupby('InvoiceDate')['TotalAmount'].sum().reset_index()
    fig = px.line(daily_sales, x='InvoiceDate', y='TotalAmount', title="Günlük Ciro Trendi")
    st.plotly_chart(fig, use_container_width=True)

with tab_mkt:
    st.header("Pazarlama & Müşteri Analizi")
    # Şehir bazlı satış dağılımı
    city_sales = filtered_df.groupby('City')['TotalAmount'].sum().reset_index()
    fig_city = px.bar(city_sales, x='City', y='TotalAmount', title="Şehir Bazlı Toplam Satış")
    st.plotly_chart(fig_city, use_container_width=True)

with tab_scm:
    st.header("Tedarik Zinciri & Stok Tahminleri")
    # Yayınevi / Ürün bazlı sipariş yoğunlukları
    top_products = filtered_df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
    fig_top = px.bar(top_products, x='Quantity', y='Description', orientation='h', title="En Çok Satan 10 Kitap")
    st.plotly_chart(fig_top, use_container_width=True)