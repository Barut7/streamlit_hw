import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI (DOĞRUDAN ÇALIŞTIRILDIĞINDA)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RafRadar | Yönetici Özeti & Veri Kalitesi",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def render_executive_summary(filtered_df=None, raw_df=None):
    # Arkadaşlarının tasarımındaki temiz, gazetecilik tarzı kart ve tipografi CSS'i
    st.markdown("""
    <style>
        /* Genel Arka Plan ve Font */
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Üst Navigasyon / Başlık Barı */
        .top-nav {
            background-color: #FFFFFF;
            padding: 12px 24px;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 8px;
        }
        .brand-title {
            font-size: 20px;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.5px;
        }
        .page-subtitle {
            font-size: 14px;
            color: #64748B;
            margin-top: -4px;
            margin-bottom: 18px;
        }

        /* KPI Kartları */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            height: 100%;
        }
        .kpi-label {
            font-size: 12px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .kpi-delta-pos {
            font-size: 12px;
            font-weight: 600;
            color: #16A34A;
            background: #DCFCE7;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .kpi-delta-neg {
            font-size: 12px;
            font-weight: 600;
            color: #DC2626;
            background: #FEE2E2;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .kpi-delta-warn {
            font-size: 12px;
            font-weight: 600;
            color: #D97706;
            background: #FEF3C7;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .kpi-subtext {
            font-size: 11px;
            color: #94A3B8;
            margin-left: 6px;
        }

        /* Numaralı Analiz Kartları (Arkadaşlarının 1., 2., 3. formatı) */
        .section-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .section-num {
            font-size: 11px;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
        }
        .section-num span {
            background-color: #E2E8F0;
            color: #0F172A;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 6px;
        }
        .section-takeaway {
            font-size: 16px;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 14px;
            line-height: 1.35;
        }
        .insight-box {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 12px 14px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            color: #334155;
            margin-top: 10px;
        }
        .action-badge-high {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }
        .action-badge-med {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }
        .dept-tag {
            background-color: #F1F5F9;
            color: #334155;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        /* Özel Tablo Stili */
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .custom-table th {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #E2E8F0;
            color: #64748B;
            font-size: 11px;
            text-transform: uppercase;
        }
        .custom-table td {
            padding: 12px;
            border-bottom: 1px solid #F1F5F9;
            color: #1E293B;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # 2. ÜST BAR VE SEKME SEÇİMİ
    # -----------------------------------------------------------------------------
    col_brand, col_nav, col_filter = st.columns([2, 5, 3])

    with col_brand:
        st.markdown("<div class='brand-title'>📊 RafRadar</div>", unsafe_allow_html=True)

    with col_nav:
        selected_tab = st.radio(
            "Sayfa Seçimi",
            ["Yönetici Özeti", "Veri Kalitesi & Metodoloji"],
            horizontal=True,
            label_visibility="collapsed",
            key="rafradar_nav_radio"
        )

    with col_filter:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            donem = st.selectbox(
                "Dönem",
                ["Son 12 Ay (2020-22)", "Son Çeyrek (Ç1'22)", "Yıl Sonu (Eyl-Ara)"],
                label_visibility="collapsed",
                key="rafradar_donem_select"
            )
        with f_col2:
            kategori = st.selectbox(
                "Ana Kategori",
                ["Tüm Kategoriler", "Sınav Hazırlık", "Roman / Edebiyat", "Çocuk & Aile", "Puzzle & Hobi"],
                label_visibility="collapsed",
                key="rafradar_kategori_select"
            )

    st.markdown("<hr style='margin: 4px 0 18px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # =============================================================================
    # SEKME 1: YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)
    # =============================================================================
    if selected_tab == "Yönetici Özeti":
        
        # Başlık ve Anahtar Soru (Arkadaşlarının formatı)
        st.markdown("### Yönetici Özeti")
        st.markdown(
            "<div class='page-subtitle'>"
            "<b>Şirket büyüyor mu, kârlılığımız güvende mi ve bu hafta hangi kritik aksiyonları almalıyız?</b> &nbsp;|&nbsp; "
            "Kapsam: Tüm B2B İşlemleri • Karşılaştırma: Önceki Dönem / Yıl"
            "</div>", 
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------------------
        # ÜST KPI KARTLARI (5 SÜTUN)
        # -------------------------------------------------------------------------
        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Yıllık Net Ciro (Son Çeyrek)</div>
                <div class="kpi-value">₺29,58 Mn</div>
                <span class="kpi-delta-pos">▲ %10,5</span>
                <span class="kpi-subtext">Ç1'22: ₺3,29 Mn net</span>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Tahmini Brüt Kâr & Marj</div>
                <div class="kpi-value">₺8,16 Mn</div>
                <span class="kpi-delta-warn">%27,6 Marj</span>
                <span class="kpi-subtext">Maliyet: Liste %44'ü</span>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Aktif Müşteri & Pareto</div>
                <div class="kpi-value">962</div>
                <span class="kpi-delta-pos">▲ +38 net</span>
                <span class="kpi-subtext">%18'i = %80 Ciro</span>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Ortalama İskonto & Kaçak</div>
                <div class="kpi-value">%38,5</div>
                <span class="kpi-delta-neg">₺189,9 bin</span>
                <span class="kpi-subtext">7.964 şüpheli sipariş</span>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Aktif SKU & Ölü Stok</div>
                <div class="kpi-value">6.418</div>
                <span class="kpi-delta-neg">1.205 Ölü SKU</span>
                <span class="kpi-subtext">180+ gün satışsız</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # -------------------------------------------------------------------------
        # ORTA BÖLÜM: 1. FİNANSAL SEZONSALLIK VE 2. KÂR SIZINTISI / FIRSAT RADARI
        # -------------------------------------------------------------------------
        col_left, col_right = st.columns([6, 4])

        with col_left:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>1</span> FİNANSAL PERFORMANS VE İSKONTO DİSİPLİNİ</div>
                <div class="section-takeaway">Yıllık cironun %44'ü Eylül–Aralık döneminde geliyor; ancak talep zirvedeyken bile iskonto oranı (%38,3) hiç düşmüyor.</div>
            """, unsafe_allow_html=True)

            # Aylık Ciro ve İskonto Grafiği
            months = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
            revenue_m = [2.15, 1.95, 2.10, 1.80, 1.55, 1.35, 1.12, 2.45, 3.40, 3.79, 4.12, 3.80]
            discount_pct = [39.2, 39.5, 38.9, 39.8, 39.9, 39.7, 39.8, 38.6, 38.4, 38.2, 38.5, 38.3]
            colors = ['#CBD5E1']*8 + ['#2563EB']*4  # Eyl-Ara vurgulu

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=months, y=revenue_m, name="Net Ciro (₺ Mn)",
                marker_color=colors, text=[f"₺{v}M" for v in revenue_m], textposition="outside"
            ))
            fig1.add_trace(go.Scatter(
                x=months, y=discount_pct, name="Ort. İskonto (%)",
                yaxis="y2", mode="lines+markers",
                line=dict(color="#DC2626", width=2.5),
                marker=dict(size=7)
            ))
            fig1.update_layout(
                height=290, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Net Ciro (₺ Mn)", range=[0, 4.8], gridcolor="#F1F5F9"),
                yaxis2=dict(title="İskonto (%)", overlaying="y", side="right", range=[30, 45], showgrid=False)
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("""
                <div class="insight-box">
                    <b>Stratejik Çıkarım:</b> Ekim zirvesi (₺3,79 Mn) ile Temmuz çukuru (₺1,12 Mn) arasında <b>3,4 kat</b> fark var. 
                    Buna rağmen yoğun sezonda (Eylül-Aralık) sabit tutulan %38,3'lük iskonto, organik talebin yüksek olduğu dönemde <b>gereksiz marj kaybı</b> yaratıyor. 
                    Ayrıca siparişlerin <b>%25'i Pazartesi</b> gününe yığılarak depo darboğazı oluşturuyor.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>2</span> KÂR SIZINTISI VE FIRSAT HAVUZU</div>
                <div class="section-takeaway">Müdahale edilmeyen 4 operasyonel sızıntı, yıllık ₺1,45 Mn'luk erişilebilir ciro ve kâr potansiyelini riske atıyor.</div>
            """, unsafe_allow_html=True)

            # Fırsat / Risk Yatay Bar Grafiği
            leak_items = [
                "Norm Dışı İskonto Sızıntısı<br>(Medyan +10 puan sapma)",
                "Ç+1 Yeni Müşteri Kaybı<br>(%56'sı 2. siparişi vermiyor)",
                "AX Sınıfı Stoksuz Kalma Riski<br>(250 kritik SKU korumasız)",
                "Yüksek Riskli VIP Müşteriler<br>(A/B segmenti 71 pasif bayi)"
            ]
            impact_values = [190, 310, 420, 530]  # Bin TL etkisi

            fig2 = go.Figure(go.Bar(
                x=impact_values, y=leak_items, orientation='h',
                marker_color=['#F59E0B', '#3B82F6', '#10B981', '#EF4444'],
                text=[f"₺{v} bin" for v in impact_values], textposition="outside"
            ))
            fig2.update_layout(
                height=290, margin=dict(l=10, r=55, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Tahmini Finansal Etki / Risk (₺ Bin)", range=[0, 650], gridcolor="#F1F5F9"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
                <div class="insight-box" style="border-left-color: #DC2626;">
                    <b>Acil Odak:</b> Tek başına <b>S00172 (₺287 bin)</b> ve <b>A00418 (₺420 bin)</b> gibi üst segment bayilerin 60-200 gündür sipariş kesmesi en büyük ciro tehdididir.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------------------
        # ALT BÖLÜM: 3. PARETO KONSANTRASYONU VE 4. YÖNETİM AKSİYON MATRİSİ
        # -------------------------------------------------------------------------
        col_b1, col_b2 = st.columns([4, 6])

        with col_b1:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>3</span> ÇİFT TARAFLI YOĞUNLAŞMA (PARETO RİSKİ)</div>
                <div class="section-takeaway">Cironun %80'i müşterilerin sadece %18'ine ve ürünlerin %9'una (742 SKU) bağımlı.</div>
            """, unsafe_allow_html=True)

            # Pareto karşılaştırma barları
            pareto_df = pd.DataFrame({
                "Kategori": ["Şampiyon & Sadık Müşteri (%35)", "Risk Altındaki Müşteriler (%15)", "A Sınıfı Ürünler - AX/AY/AZ (%9 SKU)", "C Sınıfı & Ölü Ürünler (%72 SKU)"],
                "Ciro Payı (%)": [70, 15, 80, 5],
                "Renk": ["#2563EB", "#EF4444", "#0D9488", "#94A3B8"]
            })

            fig3 = go.Figure(go.Bar(
                x=pareto_df["Ciro Payı (%)"],
                y=pareto_df["Kategori"],
                orientation='h',
                marker_color=pareto_df["Renk"],
                text=[f"%{v} Ciro" for v in pareto_df["Ciro Payı (%)"]],
                textposition="inside"
            ))
            fig3.update_layout(
                height=250, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Toplam Ciro Payı (%)", range=[0, 100], gridcolor="#F1F5F9"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("""
                <div style="font-size: 12px; color: #475569; margin-top: 6px;">
                    • <b>Müşteri Kırılganlığı:</b> İlk 10 müşteri toplam cironun <b>%41'ini</b> tek başına üretiyor.<br>
                    • <b>Stok Kırılganlığı:</b> Cironun %40'ını çeken <b>250 AX ürününde</b> sıfır stoksuzluk politikası zorunlu.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b2:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>4</span> YÖNETİM KURULU 30 GÜNLÜK AKSİYON PLANI</div>
                <div class="section-takeaway">Analiz sekmelerinden süzülen, hemen uygulanabilir ve en yüksek geri dönüşlü 4 stratejik karar:</div>
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Birim</th>
                            <th>Tespit Edilen Kök Sorun</th>
                            <th>Önerilen Somut Aksiyon</th>
                            <th>Hedef / Etki</th>
                            <th>Öncelik</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="dept-tag">Pazarlama</span></td>
                            <td><b>A00418</b> (₺420B) ve <b>S00172</b> (₺288B) dahil 7 VIP müşteri 2x süredir sipariş vermedi.</td>
                            <td>Müşteri temsilcisi ile doğrudan telefon/ziyaret ve özel geri kazanım ıskontosu.</td>
                            <td><b>₺890 bin+</b><br><span style="font-size:11px; color:#64748B;">Ciro Koruma</span></td>
                            <td><span class="action-badge-high">KRİTİK (Bu Hafta)</span></td>
                        </tr>
                        <tr>
                            <td><span class="dept-tag">Satış & Fiyat</span></td>
                            <td>7.964 siparişte kategori medyanından <b>10+ puan</b> fazla iskonto (Sözlük/Kişisel Gel.).</td>
                            <td>ERP'de kategori bazlı otomatik iskonto tavanı (%47 üst sınır) ve yönetici onayı.</td>
                            <td><b>₺189,9 bin</b><br><span style="font-size:11px; color:#64748B;">Doğrudan Kâr</span></td>
                            <td><span class="action-badge-high">KRİTİK (Bu Hafta)</span></td>
                        </tr>
                        <tr>
                            <td><span class="dept-tag">Tedarik</span></td>
                            <td>Eylül Sınav Hazırlık zirvesi (1,7x talep) öncesi <b>AX sınıfı 250 SKU</b> stok riski.</td>
                            <td>Holt-Winters (WAPE %9,4) ile otomatik ROP emri; 1.205 ölü SKU için eritme paketi.</td>
                            <td><b>0 Stoksuz Gün</b><br><span style="font-size:11px; color:#64748B;">%95 Hizmet Seviyesi</span></td>
                            <td><span class="action-badge-med">YÜKSEK (15 Gün)</span></td>
                        </tr>
                        <tr>
                            <td><span class="dept-tag">CRM / Satış</span></td>
                            <td>Yeni müşterilerin <b>%56'sı</b> ilk siparişten sonraki çeyrekte (Ç+1) kaybediliyor.</td>
                            <td>İlk faturadan 30-45 gün sonra otomatik "Birlikte Alınan" (Puzzle+Çocuk 2,1x) teklifi.</td>
                            <td><b>%44 → %50</b><br><span style="font-size:11px; color:#64748B;">Ç+1 Elde Tutma</span></td>
                            <td><span class="action-badge-med">ORTA (30 Gün)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # =============================================================================
    # SEKME 2: VERİ KALİTESİ & METODOLOJİ (DATA QUALITY & GOVERNANCE)
    # =============================================================================
    else:
        st.markdown("### Veri Kalitesi & Metodoloji Karnesi")
        st.markdown(
            "<div class='page-subtitle'>"
            "<b>Analiz ve simülasyon sonuçlarına ne kadar güvenebiliriz?</b> &nbsp;|&nbsp; "
            "Veri setindeki yapısal kısıtlar, anomali taraması ve model doğrulama metrikleri"
            "</div>", 
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------------------
        # ÜST KPI KARTLARI (VERİ SAĞLIĞI)
        # -------------------------------------------------------------------------
        dq1, dq2, dq3, dq4 = st.columns(4)

        with dq1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Veri Tamlık & Bütünlük Skoru</div>
                <div class="kpi-value">%94,2</div>
                <span class="kpi-delta-pos">Güvenilir</span>
                <span class="kpi-subtext">Fatura, SKU ve Bayi ID tam</span>
            </div>
            """, unsafe_allow_html=True)

        with dq2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Tespit Edilen Fiyat/İskonto Sapması</div>
                <div class="kpi-value">7.964</div>
                <span class="kpi-delta-warn">Anomali</span>
                <span class="kpi-subtext">Medyan +10 puan üstü satır</span>
            </div>
            """, unsafe_allow_html=True)

        with dq3:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Kayıp / Eksik Sütun Sayısı</div>
                <div class="kpi-value">3 Değişken</div>
                <span class="kpi-delta-neg">Simüle Edildi</span>
                <span class="kpi-subtext">Stok, Alış Maliyeti, Saat</span>
            </div>
            """, unsafe_allow_html=True)

        with dq4:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Tahmin & Churn Model Başarısı</div>
                <div class="kpi-value">0,89 AUC</div>
                <span class="kpi-delta-pos">WAPE %9,4</span>
                <span class="kpi-subtext">GBM & Holt-Winters Onaylı</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # -------------------------------------------------------------------------
        # 1. VERİ KISITLARI VE VARSAYIMLAR TABLOSU & 2. ANOMALİ DAĞILIMI
        # -------------------------------------------------------------------------
        col_dq1, col_dq2 = st.columns([6, 4])

        with col_dq1:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>1</span> VERİ KISITLARI VE ANALİTİK VARSAYIMLAR ŞEFFAFLIK TABLOSU</div>
                <div class="section-takeaway">Ham B2B işlem verisinde yer almayan 4 operasyonel parametre, bilimsel varsayımlar ve simülasyonla tamamlanmıştır.</div>
                <table class="custom-table">
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
                            <td><span class="dept-tag">Tedarik Zinciri</span></td>
                            <td>Stok planı, geçmiş haftalık talep varyansı (CV), kullanıcı tedarik süresi (L=2 hafta) ve %95 hizmet seviyesiyle ($SS = z \\cdot \\sigma \\cdot \\sqrt{L}$) simüle edildi.</td>
                            <td><span class="action-badge-med">Orta (Kullanıcı Girdili)</span></td>
                        </tr>
                        <tr>
                            <td><b>Ürün Alış / Üretim Maliyeti</b></td>
                            <td><span class="dept-tag">Satış & Fiyat</span></td>
                            <td>Kâr simülasyonlarında ürün alış maliyeti, sektör standardı olan <b>Liste Fiyatının %44'ü</b> olarak sabitlendi.</td>
                            <td><span class="action-badge-med">Orta (Sabit Marj)</span></td>
                        </tr>
                        <tr>
                            <td><b>Sipariş Saat Damgası (Timestamp)</b></td>
                            <td><span class="dept-tag">Satış & Fiyat</span></td>
                            <td>Veride yalnızca gün bilgisi olduğu için saatlik ısı haritası yerine <b>Gün x Kategori</b> yoğunluk matrisi kullanıldı (%25 Pazartesi yığılması).</td>
                            <td><span class="kpi-delta-pos">Yüksek (Tam Veri)</span></td>
                        </tr>
                        <tr>
                            <td><b>Yüksek WAPE'li (>%20) Düzensiz SKU'lar</b></td>
                            <td><span class="dept-tag">Tahminleme</span></td>
                            <td>CV > 1 olan Z sınıfı ürünlerde (%38 SKU) istatistiksel tahmin yerine <b>"Siparişe Göre Tedarik (MTO)"</b> politikası atandı.</td>
                            <td><span class="kpi-delta-pos">Yüksek (Risk Korumalı)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_dq2:
            st.markdown("""
            <div class="section-card">
                <div class="section-num"><span>2</span> VERİ ANOMALİLERİ VE UÇ DEĞER TARAMASI</div>
                <div class="section-takeaway">İskonto sapmaları ve hareketsiz SKU'lar veri dağılımını çarpıtıyor; modellerde kırpma (winsorizing) uygulandı.</div>
            """, unsafe_allow_html=True)

            anom_cats = ["Sözlük (Medyan %38 → Max %55)", "Bulmaca (Medyan %45 → Max %55)", "Kişisel Gel. (Medyan %38 → Max %50)", "Roman/Öykü (Medyan %39 → Max %48)"]
            anom_loss = [78.4, 45.2, 38.1, 28.2]

            fig_dq = go.Figure(go.Bar(
                x=anom_loss, y=anom_cats, orientation='h',
                marker_color='#DC2626',
                text=[f"₺{v}B Kayıp" for v in anom_loss], textposition="outside"
            ))
            fig_dq.update_layout(
                height=230, margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Kategori Bazlı Norm Dışı İskonto Tutarı (₺ Bin)", range=[0, 100], gridcolor="#F1F5F9"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_dq, use_container_width=True)

            st.markdown("""
                <div class="insight-box">
                    <b>Veri Temizliği Notu:</b> <code>DPF...0948</code> ve <code>DPF...0962</code> gibi faturalarda görülen +%17 puanlık iskonto sapmaları, fiyat elastikiyeti modelini bozmaması için eğitim setinde <b>%95 persentilde (%47-50)</b> sınırlandırılmıştır.
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_executive_summary()
