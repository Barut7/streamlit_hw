import os
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# 1. SAYFA AYARLARI (SOL MENÜ AÇIK - RAFRADAR V2 ORTAK YAPI)
# =============================================================================
st.set_page_config(
    page_title="RafRadar · B2B Karar Destek Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_shared_rafradar_module():
    """
    'yönetici özeti.py' içerisindeki ortak RafRadar v2 tasarım, sol menü,
    üst KPI şeridi, sohbet asistanı ve veri motorunu modül olarak yükler.
    """
    exec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yönetici özeti.py")
    if not os.path.exists(exec_path):
        return None

    orig_set_page_config = st.set_page_config
    try:
        st.set_page_config = lambda *args, **kwargs: None
        spec = importlib.util.spec_from_file_location("yonetici_ozeti_shared", exec_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        st.set_page_config = orig_set_page_config


def try_run_external_department_file(candidate_filenames: list, entry_functions: list, ctx: dict, yo_mod) -> bool:
    """
    Diğer departmanlardaki ekip arkadaşları kendi kodlarını ayrı bir .py dosyası olarak
    (örneğin 'satis.py', 'pazarlama.py', 'tedarik.py') eklediğinde otomatik algılar ve
    ortak filtre/veri bağlamı (ctx) ile çalıştırır.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in candidate_filenames:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            orig_set_page_config = st.set_page_config
            try:
                st.set_page_config = lambda *args, **kwargs: None
                spec = importlib.util.spec_from_file_location(f"dept_{fname}", fpath)
                mod = importlib.util.module_from_spec(spec)
                # Modül içine ortak filtreleri ve yardımcı fonksiyonları enjekte et
                mod.RAFRADAR_CTX = ctx
                mod.RAFRADAR_SHARED = yo_mod
                spec.loader.exec_module(mod)
                for fn_name in entry_functions:
                    if hasattr(mod, fn_name):
                        fn = getattr(mod, fn_name)
                        try:
                            fn(ctx)
                        except TypeError:
                            fn(ctx["sel_df"], ctx["full_raw_df"])
                        return True
                return True
            finally:
                st.set_page_config = orig_set_page_config
    return False


# =============================================================================
# 2. DEPARTMAN ŞABLONLARI (ORTAK SOL MENÜ & ÜST BAR İLE %100 UYUMLU)
# =============================================================================
def render_satis_department(ctx: dict, yo):
    """
    📈 Satış & Fiyat Departmanı Sayfası
    Ortak üst bar (head + pills + 5'li kcard KPI şeridi) + numaralı analiz kartları.
    Ekip arkadaşları kendi grafiklerini doğrudan bu fonksiyona veya 'satis.py' dosyasına ekleyebilir.
    """
    if try_run_external_department_file(
        ["satis.py", "satış.py", "satis_fiyat.py"],
        ["render_satis_page", "render_satis", "render_page", "main"],
        ctx,
        yo,
    ):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]
    comp_df = ctx["comp_df"]

    disc_pts_str = f"{m['disc_diff_pts']:+.1f} puan".replace(".", ",")
    disc_cls = "neg" if m["disc_diff_pts"] > 0.5 else ("pos" if m["disc_diff_pts"] < -0.5 else "mute")

    kpi_cards = [
        {
            "label": "Net ciro",
            "main_val": yo.fmt_tl(S["net"]),
            "main_chg_html": yo.chg_badge_html(m["rev_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": yo.fmt_tl(S["dnet"], 1),
            "sub_chg_html": yo.chg_badge_html(m["d_rev_chg"]),
            "top_color": "#3d9a84",
        },
        {
            "label": "Brüt (liste) ciro",
            "main_val": yo.fmt_tl(S["gross"]),
            "main_chg_html": f'<span class="chg mute">{yo.fmt_int(S["qty"])} adet</span>',
            "sub_label": "Günlük ort. adet",
            "sub_val": yo.fmt_int(S["dqty"]),
            "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]),
            "top_color": "#1f2a44",
        },
        {
            "label": "Ortalama indirim",
            "main_val": f"%{S['disc']:.1f}".replace(".", ","),
            "main_chg_html": f'<span class="chg {disc_cls}">{disc_pts_str}</span>',
            "sub_label": "Fazla indirim kaçağı",
            "sub_val": yo.fmt_tl(m["discount_leak_tl"], 1),
            "sub_chg_html": f'<span class="chg neg">{yo.fmt_int(m["suspicious_orders_count"])} satır</span>',
            "top_color": "#d0574f",
        },
        {
            "label": "Sipariş sayısı",
            "main_val": yo.fmt_int(S["orders"]),
            "main_chg_html": yo.chg_badge_html(m["inv_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": f"{S['dorders']:.1f}".replace(".", ","),
            "sub_chg_html": yo.chg_badge_html(m["d_inv_chg"]),
            "top_color": "#3b6fd1",
        },
        {
            "label": "Sepet ortalaması",
            "main_val": yo.fmt_tl(S["aov"], 1),
            "main_chg_html": yo.chg_badge_html(C["aov_chg"]),
            "sub_label": "Karşılaştırma",
            "sub_val": yo.fmt_tl(C["aov"], 1),
            "sub_chg_html": '<span class="chg mute">sipariş başı</span>',
            "top_color": "#e0a526",
        },
    ]

    yo.render_department_header_and_kpis(
        title="Satış & Fiyat",
        subtitle="Hangi kategori, şehir ve indirim politikası ciroyu büyütüyor?",
        ctx=ctx,
        kpi_cards=kpi_cards,
        note_text="Tüm satış, fiyat ve iskonto panelleri sol menüdeki <b>Seçili dönem</b> ve <b>Karşılaştırma dönemi</b> filtrelerine göre canlı güncellenir.",
    )

    c1, c2 = st.columns([5.6, 4.4])
    with c1:
        yo.render_section_card_header(
            1,
            "Günlük / Aylık Ciro ve İndirim Trendi",
            f"Seçili dönem ({ctx['sel_lbl']}) satış seyri",
            f"Seçili dönemde toplam {yo.fmt_tl(S['net'])} net ciro ve ortalama %{S['disc']:.1f} iskonto gerçekleşti.",
        )
        daily_df = (
            sel_df.groupby("Date", as_index=False)
            .agg(NetRev=("TotalAmount", "sum"), GrossRev=("GrossAmount", "sum"), Orders=("InvoiceNo", "nunique"))
            .sort_values("Date")
        )
        fig_s1 = go.Figure()
        fig_s1.add_trace(
            go.Bar(
                x=daily_df["Date"],
                y=daily_df["NetRev"],
                name="Net Ciro (₺)",
                marker_color="#1f2a44",
            )
        )
        fig_s1.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#1f2a44"),
            xaxis=dict(gridcolor="#efe9de"),
            yaxis=dict(title="Net Ciro (₺)", gridcolor="#efe9de"),
        )
        st.plotly_chart(fig_s1, use_container_width=True)

    with c2:
        yo.render_section_card_header(
            2,
            "En Çok Ciro Üreten İller ve Kategoriler",
            "Seçili dönem vs karşılaştırma dönemi il dağılımı",
            f"Lider il {m['top_city_name']} (%{m['top_city_share']:.1f} pay), lider kategori {m['top_cat_name']} (%{m['top_cat_share']:.1f} pay).",
        )
        city_df = (
            sel_df.groupby("City", as_index=False)["TotalAmount"]
            .sum()
            .nlargest(10, "TotalAmount")
            .sort_values("TotalAmount", ascending=True)
        )
        fig_s2 = go.Figure(
            go.Bar(
                x=city_df["TotalAmount"],
                y=city_df["City"],
                orientation="h",
                marker_color="#3d9a84",
                text=[yo.fmt_tl(v, 1) for v in city_df["TotalAmount"]],
                textposition="outside",
            )
        )
        fig_s2.update_layout(
            height=290,
            margin=dict(l=10, r=65, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#1f2a44"),
            xaxis=dict(
                title="Net Ciro (₺)",
                range=[0, float(city_df["TotalAmount"].max()) * 1.28 if not city_df.empty else 1],
                gridcolor="#efe9de",
            ),
        )
        st.plotly_chart(fig_s2, use_container_width=True)

    with st.expander("🔌 Satış & Fiyat Departmanı Kod Entegrasyon Rehberi"):
        st.markdown(
            """
            **Satış departmanı kodlarını entegre etmek için 2 yöntem:**
            1. Klasöre **`satis.py`** dosyası ekleyip içine `def render_satis_page(ctx):` fonksiyonu yazabilirsiniz (sistem otomatik algılar ve çalıştırır).
            2. Veya doğrudan `streamlit_app.py` içindeki `render_satis_department(ctx, yo)` fonksiyonunun içine grafiklerinizi ekleyebilirsiniz.
            - `ctx["sel_df"]`: Seçili dönem filtrelenmiş DataFrame (`S`)
            - `ctx["comp_df"]`: Karşılaştırma dönemi filtrelenmiş DataFrame (`C`)
            - `ctx["S"]` ve `ctx["C"]`: Hazır hesaplanmış özet metrikler (`net`, `dnet`, `disc`, `orders`, `qty`, `aov`)
            """
        )


def render_pazarlama_department(ctx: dict, yo):
    """
    🎯 Pazarlama & Müşteri Departmanı Sayfası
    rafradar-versiyon2.html satır 800-808'deki 'Pazarlama & Müşteri' üst barı ve 5'li KPI şeridi ile birebir uyumlu.
    """
    if try_run_external_department_file(
        ["pazarlama.py", "musteri.py", "pazarlama_musteri.py"],
        ["render_pazarlama_page", "render_pazarlama", "render_page", "main"],
        ctx,
        yo,
    ):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]

    # rafradar-versiyon2.html satır 802-807'deki 5 KPI kartı:
    # Aktif müşteri | Yeni müşteri | Satılan adet | Net ciro | Müşteri başı ciro
    kpi_cards = [
        {
            "label": "Aktif müşteri",
            "main_val": yo.fmt_int(S["cust"]),
            "main_chg_html": yo.chg_badge_html(m["cust_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": f"{S['dcust']:.1f}".replace(".", ","),
            "sub_chg_html": yo.chg_badge_html(m["d_cust_chg"]),
            "top_color": "#3d9a84",
        },
        {
            "label": "Yeni müşteri",
            "main_val": yo.fmt_int(S["new_cust"]),
            "main_chg_html": '<span class="chg mute">ilk alımı bu dönemde</span>',
            "sub_label": "Günlük ort.",
            "sub_val": f"{S['new_cust'] / max(1, S['days']):.2f}".replace(".", ","),
            "sub_chg_html": "",
            "top_color": "#3b6fd1",
        },
        {
            "label": "Satılan adet",
            "main_val": f"{S['qty'] / 1000.0:,.1f} bin".replace(",", "X").replace(".", ",").replace("X", "."),
            "main_chg_html": yo.chg_badge_html(C["qty_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": yo.fmt_int(S["dqty"]),
            "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]),
            "top_color": "#1f2a44",
        },
        {
            "label": "Net ciro",
            "main_val": yo.fmt_tl(S["net"]),
            "main_chg_html": yo.chg_badge_html(m["rev_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": yo.fmt_tl(S["dnet"], 1),
            "sub_chg_html": yo.chg_badge_html(m["d_rev_chg"]),
            "top_color": "#3d9a84",
        },
        {
            "label": "Müşteri başı ciro",
            "main_val": yo.fmt_tl(S["per_cust"], 1),
            "main_chg_html": yo.chg_badge_html(C["per_cust_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": yo.fmt_tl(S["d_per_cust"], 0),
            "sub_chg_html": yo.chg_badge_html(C["d_per_cust_chg"]),
            "top_color": "#e0a526",
        },
    ]

    yo.render_department_header_and_kpis(
        title="Pazarlama & Müşteri",
        subtitle="Hangi müşteriye, hangi ürünle, ne zaman gitmeliyiz?",
        ctx=ctx,
        kpi_cards=kpi_cards,
        note_text="Tüm paneller sol menüde seçtiğin tarihlere göre hesaplanır (Seçili dönem vs Karşılaştırma dönemi).",
    )

    c1, c2 = st.columns([5.2, 4.8])
    with c1:
        net_diff_cust = S["cust"] - C["cust"]
        yo.render_section_card_header(
            1,
            "Müşteri tabanı bu dönemde nasıl değişti?",
            f"Karşılaştırma dönemindeki aktif müşteri {yo.fmt_int(C['cust'])} → seçili dönemdeki aktif müşteri {yo.fmt_int(S['cust'])}",
            f"Bu dönemde {yo.fmt_int(S['new_cust'])} yeni müşteri kazanıldı; net müşteri değişimi {net_diff_cust:+d} oldu.",
        )
        cust_seg_df = pd.DataFrame(
            {
                "Segment": ["A · VIP Müşteriler (%80 Ciro)", "B · Gelişen Müşteriler (%15 Ciro)", "C · Küçük Alıcılar (%5 Ciro)"],
                "Müşteri Sayısı": [m["vip_cust_count"], m["mid_cust_count"], m["tail_cust_count"]],
                "Ciro (₺)": [m["vip_rev_sum"], m["mid_rev_sum"], m["tail_rev_sum"]],
            }
        )
        fig_m1 = go.Figure(
            go.Bar(
                x=cust_seg_df["Segment"],
                y=cust_seg_df["Müşteri Sayısı"],
                marker_color=["#1f2a44", "#3d9a84", "#c9c2b4"],
                text=[f"{yo.fmt_int(v)} müşteri" for v in cust_seg_df["Müşteri Sayısı"]],
                textposition="outside",
            )
        )
        fig_m1.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=15, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#1f2a44"),
            yaxis=dict(
                title="Müşteri Sayısı",
                range=[0, float(cust_seg_df["Müşteri Sayısı"].max()) * 1.25 if not cust_seg_df.empty else 1],
                gridcolor="#efe9de",
            ),
        )
        st.plotly_chart(fig_m1, use_container_width=True)

    with c2:
        yo.render_section_card_header(
            2,
            "Kayıp (Churn) Riskindeki En Büyük VIP Müşteriler",
            f"{m['recency_threshold']}+ gündür sipariş vermeyen A sınıfı müşteriler",
            f"Toplam {yo.fmt_int(m['at_risk_vip_count'])} VIP müşteride gecikme tespit edildi (korunabilir ciro: +{yo.fmt_tl(m['recoverable_vip_tl'])}).",
        )
        top_cust = (
            sel_df.groupby("CustomerID", as_index=False)
            .agg(
                Ciro=("TotalAmount", "sum"),
                Siparis=("InvoiceNo", "nunique"),
                SonAlim=("InvoiceDate", "max"),
            )
            .nlargest(8, "Ciro")
        )
        top_cust["Ciro"] = top_cust["Ciro"].round(0)
        st.dataframe(
            top_cust.rename(
                columns={
                    "CustomerID": "Müşteri No",
                    "Ciro": "Seçili Dönem Cirosu (₺)",
                    "Siparis": "Sipariş Sayısı",
                    "SonAlim": "Son Fatura Tarihi",
                }
            ),
            use_container_width=True,
            height=265,
        )

    with st.expander("🔌 Pazarlama & Müşteri Departmanı Kod Entegrasyon Rehberi"):
        st.markdown(
            """
            **Pazarlama departmanı kodlarını entegre etmek için 2 yöntem:**
            1. Klasöre **`pazarlama.py`** dosyası ekleyip içine `def render_pazarlama_page(ctx):` fonksiyonu yazabilirsiniz.
            2. Veya doğrudan `streamlit_app.py` içindeki `render_pazarlama_department(ctx, yo)` fonksiyonuna RFM / Cohort / Churn grafiklerinizi ekleyebilirsiniz.
            """
        )


def render_tedarik_department(ctx: dict, yo):
    """
    🚚 Tedarik Zinciri Departmanı Sayfası
    rafradar-versiyon2.html Tedarik Zinciri yapısı (ABC/XYZ, hareketsiz stok, emniyet stoğu ve sipariş noktası tablosu) ile uyumlu.
    """
    if try_run_external_department_file(
        ["tedarik.py", "stok.py", "tedarik_zinciri.py"],
        ["render_tedarik_page", "render_tedarik", "render_page", "main"],
        ctx,
        yo,
    ):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]

    kpi_cards = [
        {
            "label": "Aktif ürün (SKU)",
            "main_val": yo.fmt_int(S["sku"]),
            "main_chg_html": yo.chg_badge_html(m["sku_chg"]),
            "sub_label": "A sınıfı yıldız ürün",
            "sub_val": f"{yo.fmt_int(m['a_sku_count'])} ürün",
            "sub_chg_html": '<span class="chg pos">=%80 ciro</span>',
            "top_color": "#1f2a44",
        },
        {
            "label": "Satılan toplam adet",
            "main_val": yo.fmt_int(S["qty"]),
            "main_chg_html": yo.chg_badge_html(C["qty_chg"]),
            "sub_label": "Günlük ort.",
            "sub_val": yo.fmt_int(S["dqty"]),
            "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]),
            "top_color": "#3d9a84",
        },
        {
            "label": f"Satışsız ürün ({m['dead_threshold']}+ gün)",
            "main_val": yo.fmt_int(m["dead_sku_count"]),
            "main_chg_html": '<span class="chg neg">hareketsiz</span>',
            "sub_label": "Yavaş dönen C sınıfı",
            "sub_val": f"{yo.fmt_int(m['c_sku_count'])} ürün",
            "sub_chg_html": '<span class="chg mute">%5 ciro payı</span>',
            "top_color": "#d0574f",
        },
        {
            "label": "Stoksuz kalma riski",
            "main_val": yo.fmt_tl(m["stockout_risk_tl"], 1),
            "main_chg_html": '<span class="chg neg">korunmalı</span>',
            "sub_label": "Hedef hizmet seviyesi",
            "sub_val": "%95 (z=1,65)",
            "sub_chg_html": '<span class="chg pos">emniyet stoğu</span>',
            "top_color": "#e0a526",
        },
        {
            "label": "Sipariş sıklığı",
            "main_val": yo.fmt_int(S["orders"]),
            "main_chg_html": yo.chg_badge_html(m["inv_chg"]),
            "sub_label": "Günlük ort. sipariş",
            "sub_val": f"{S['dorders']:.1f}".replace(".", ","),
            "sub_chg_html": yo.chg_badge_html(m["d_inv_chg"]),
            "top_color": "#3b6fd1",
        },
    ]

    yo.render_department_header_and_kpis(
        title="Tedarik Zinciri",
        subtitle="Hangi üründen, ne zaman, ne kadar sipariş vermeliyiz?",
        ctx=ctx,
        kpi_cards=kpi_cards,
        note_text="Emniyet stoğu ve sipariş noktası (ROP) hesaplamaları aşağıdan seçtiğiniz tedarik süresi (hafta) ve hizmet seviyesine (%) göre anında güncellenir.",
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        lt_weeks = st.slider("⏱️ Tedarik Süresi (Hafta - L)", min_value=1, max_value=6, value=2, key="rr_scm_lt")
    with p_col2:
        sl_choice = st.select_slider(
            "🛡️ Hedef Hizmet Seviyesi (%)",
            options=[90, 95, 98, 99],
            value=95,
            key="rr_scm_sl",
        )
    z_map = {90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}
    z_val = z_map.get(sl_choice, 1.65)

    yo.render_section_card_header(
        1,
        "En Çok Satan 12 Ürün İçin Stok Planı (Güvenlik Stoğu & Sipariş Noktası)",
        f"Tedarik süresi {lt_weeks} hafta, hizmet seviyesi %{sl_choice} (z={z_val}) ile önerilen stok politikası",
        f"A sınıfı {yo.fmt_int(m['a_sku_count'])} ürün toplam cironun %80'ini üretiyor; bu ürünlerde stoksuz kalma maliyeti {yo.fmt_tl(m['stockout_risk_tl'])}'dir.",
    )

    top12 = (
        sel_df.groupby(["Description", "Category"], as_index=False)
        .agg(TotalQty=("Quantity", "sum"), TotalRev=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"))
        .nlargest(12, "TotalRev")
    )
    weeks_in_sel = max(1.0, ctx["s_days"] / 7.0)
    top12["WeeklyAvg"] = top12["TotalQty"] / weeks_in_sel
    top12["Demand4W"] = (top12["WeeklyAvg"] * 4.0).round(0)
    top12["SigmaW"] = (top12["WeeklyAvg"] * 0.32).clip(lower=1.0)
    top12["SafetyStock"] = (z_val * top12["SigmaW"] * np.sqrt(lt_weeks)).round(0)
    top12["ReorderPoint"] = (top12["WeeklyAvg"] * lt_weeks + top12["SafetyStock"]).round(0)
    top12["Sınıf"] = "A · Yıldız"
    top12["İzlenecek Politika"] = f"ROP ≤ {lt_weeks}hf talep + SS olduğunda otomatik sipariş"

    st.dataframe(
        top12[
            [
                "Description",
                "Category",
                "Sınıf",
                "Demand4W",
                "SafetyStock",
                "ReorderPoint",
                "TotalRev",
                "İzlenecek Politika",
            ]
        ].rename(
            columns={
                "Description": "Ürün Adı",
                "Category": "Kategori",
                "Demand4W": "4 Haftalık Talep (Adet)",
                "SafetyStock": "Güvenlik Stoğu (SS)",
                "ReorderPoint": "Sipariş Noktası (ROP)",
                "TotalRev": "Dönem Cirosu (₺)",
            }
        ),
        use_container_width=True,
    )

    with st.expander("🔌 Tedarik Zinciri Departmanı Kod Entegrasyon Rehberi"):
        st.markdown(
            """
            **Tedarik Zinciri departmanı kodlarını entegre etmek için 2 yöntem:**
            1. Klasöre **`tedarik.py`** dosyası ekleyip içine `def render_tedarik_page(ctx):` fonksiyonu yazabilirsiniz.
            2. Veya doğrudan `streamlit_app.py` içindeki `render_tedarik_department(ctx, yo)` fonksiyonuna kendi talep tahmin / ABC-XYZ grafiklerinizi ekleyebilirsiniz.
            """
        )


# =============================================================================
# 3. ANA UYGULAMA ÇATISI (ORTAK SOL MENÜ + DEPARTMAN YÖNLENDİRİCİ)
# =============================================================================
yo_mod = load_shared_rafradar_module()

if yo_mod is None:
    st.error("'yönetici özeti.py' dosyası bulunamadı!")
else:
    ctx = yo_mod.build_shared_context(pages=yo_mod.DEPARTMENT_PAGES)

    if not ctx:
        st.error("Veri seti bulunamadı! Lütfen 'B2B_Transaction_Data_.parquet' dosyasını kontrol edin.")
    elif ctx.get("empty", False):
        st.warning(
            "Seçilen tarih aralığı, kategori ve il filtresi için satış kaydı bulunamadı. "
            "Lütfen sol menüden tarih aralığını genişletin veya 'Sıfırla' butonuna tıklayın."
        )
    else:
        active_dept = ctx["page"]

        if active_dept in ("📊 Yönetici Özeti", "🛠️ Veri Kalitesi"):
            target_subview = "🛠️ Veri Kalitesi" if active_dept == "🛠️ Veri Kalitesi" else "🏛️ Yönetici Özeti"
            st.session_state["rr_top_subview"] = target_subview
            ext_nav = dict(ctx["nav"])
            ext_nav["page"] = "Veri Kalitesi" if active_dept == "🛠️ Veri Kalitesi" else "Yönetici Özeti"
            yo_mod.render_executive_summary(
                raw_df=ctx["full_raw_df"],
                external_filters=ext_nav,
            )
        elif active_dept == "📈 Satış & Fiyat":
            render_satis_department(ctx, yo_mod)
        elif active_dept == "🎯 Pazarlama & Müşteri":
            render_pazarlama_department(ctx, yo_mod)
        elif active_dept == "🚚 Tedarik Zinciri":
            render_tedarik_department(ctx, yo_mod)
