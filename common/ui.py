"""RafRadar ortak arayüz (UI), biçimlendirme, tema CSS ve HTML birebir kenar çubuğu (sidebar) bileşenleri."""
from __future__ import annotations

import datetime
import io
import math
import numpy as np
import pandas as pd
import streamlit as st

RENK = {
    "yesil": "#3d9a84",
    "kirmizi": "#d0574f",
    "sari": "#e0a526",
    "lacivert": "#1f2a44",
    "turuncu": "#d9622b",
    "mavi": "#3b6fd1",
}

AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]

SAYFALAR_META = [
    ("yonetici", "Yönetici Özeti", "#1f2a44", "🧭"),
    ("satis", "Satış & Fiyat", "#3d9a84", "💰"),
    ("pazarlama", "Pazarlama & Müşteri", "#d9622b", "🎯"),
    ("tedarik", "Tedarik Zinciri", "#3b6fd1", "📦"),
    ("kalite", "Veri Kalitesi", "#e0a526", "🔍"),
]

ROLLER_META = [
    ("yonetici", "Yönetici"),
    ("satis", "Satış"),
    ("pazarlama", "Pazarlama"),
    ("tedarik", "Tedarik Zinciri"),
]


def sayi(val, dec: int = 0) -> str:
    """Türkçe sayı biçimlendirme (binlik nokta, ondalık virgül)."""
    if val is None or (isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val))):
        return "0"
    try:
        v = float(val)
    except Exception:
        return str(val)
    if dec <= 0:
        return f"{int(round(v)):,}".replace(",", ".")
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def tl(val) -> str:
    """Türkçe para birimi biçimlendirme."""
    if val is None or (isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val))):
        return "0 ₺"
    v = float(val)
    isaret_str = "−" if v < 0 else ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{isaret_str}{sayi(av / 1_000_000, 2)} M ₺"
    return f"{isaret_str}{sayi(av, 0)} ₺"


def isaret(val, dec: int = 1) -> str:
    """Artı veya eksi işaretli sayı."""
    if val is None or (isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val))):
        return "+0"
    v = float(val)
    on = "+" if v >= 0 else "−"
    return f"{on}{sayi(abs(v), dec)}"


def _to_date(d) -> datetime.date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return pd.Timestamp(d).date()


def tarih_etiketi(a, b) -> str:
    """Tarih veya tarih aralığını Türkçe kısa biçimde döndürür."""
    da, db = _to_date(a), _to_date(b)
    if da == db:
        return f"{da.day} {AYLAR[da.month - 1]} {da.year}"
    if da.year == db.year:
        return f"{da.day} {AYLAR[da.month - 1]} – {db.day} {AYLAR[db.month - 1]} {db.year}"
    return f"{da.day} {AYLAR[da.month - 1]} {da.year} – {db.day} {AYLAR[db.month - 1]} {db.year}"


def _inject_css(aktif_sayfa: str = ""):
    aktif_renk = dict((k, c) for k, _, c, _ in SAYFALAR_META).get(aktif_sayfa, "#1f2a44")
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        :root {{
            --bg: #f6f3ee;
            --fg: #1f2a44;
            --muted: #5b6475;
            --card: #ffffff;
            --side: #faf7f2;
            --line: #e4dfd6;
            --green: #3d9a84;
            --amber: #e0a526;
            --red: #d0574f;
            --orange: #d9622b;
            --blue: #3b6fd1;
        }}
        .stApp {{
            background-color: var(--bg) !important;
            color: var(--fg);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        /* Varsayılan Streamlit sayfa menüsünü gizle (kendi HTML birebir menümüzü kullanıyoruz) */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 4.5rem !important;
            max-width: 1440px !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #faf7f2 !important;
            border-right: 1px solid #e4dfd6 !important;
            min-width: 268px !important;
            max-width: 282px !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding: 1.1rem 0.9rem 1.5rem 0.9rem !important;
        }}
        /* Sol menüdeki sayfa butonları (büyütülmüş punto ve temsili ikonlar) */
        section[data-testid="stSidebar"] div.stButton > button {{
            width: 100%;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 8px 11px !important;
            min-height: 40px !important;
            border-radius: 9px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: #1f2a44 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 14.5px !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            margin: 2px 0 !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button:hover {{
            background: rgba(31, 42, 68, 0.06) !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background: #ffffff !important;
            font-weight: 600 !important;
            border: 1px solid #e4dfd6 !important;
            box-shadow: inset 4px 0 0 {aktif_renk} !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button p {{
            font-size: 14.5px !important;
            color: #1f2a44 !important;
            margin: 0 !important;
            line-height: 1.3 !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p {{
            font-size: 15px !important;
            font-weight: 600 !important;
        }}
        /* Sol menü selectbox ve date_input kutularını HTML tasarımla eşitle */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
            background-color: #ffffff !important;
            border: 1px solid #cfc9bd !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            min-height: 34px !important;
        }}
        /* Tarih kutularındaki (stDateInput) sayı puntolarını küçült ve iç boşluğu daralt ki tam sığsın */
        section[data-testid="stSidebar"] [data-testid="stDateInput"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] [data-testid="stDateInput"] div[data-baseweb="base-input"] {{
            padding: 0 4px !important;
            min-height: 31px !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stDateInput"] input {{
            font-size: 11px !important;
            padding: 3px 2px !important;
            letter-spacing: -0.02em !important;
            text-align: center !important;
            color: #1f2a44 !important;
            font-weight: 500 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
            gap: 6px !important;
        }}
        section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {{
            font-size: 12px !important;
            color: #5b6475 !important;
            font-weight: 400 !important;
        }}
        .rr-side-hdr {{
            font-size: 11px;
            font-weight: 600;
            color: #5b6475;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 10px 0 4px 0;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #ffffff !important;
            border: 1px solid #e4dfd6 !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 2px rgba(31, 42, 68, 0.02);
        }}
        h1, h2, h3, .h {{
            font-family: 'Poppins', sans-serif !important;
            color: #1f2a44;
        }}
        .rr-page-header {{
            margin-bottom: 10px;
        }}
        .rr-page-header h1 {{
            font-size: 27px !important;
            font-weight: 700 !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            color: #1f2a44 !important;
        }}
        .rr-page-header p {{
            margin: 4px 0 0 0 !important;
            font-size: 14px !important;
            font-style: italic;
            color: #5b6475 !important;
        }}
        .h {{
            font-weight: 600;
            font-size: 15px;
            margin: 0;
            display: flex;
            align-items: center;
            color: #1f2a44;
        }}
        .sub {{
            font-size: 12px;
            color: #5b6475;
            margin: 2px 0 6px 0;
            line-height: 1.4;
        }}
        .num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 21px;
            height: 21px;
            border-radius: 50%;
            background: #1f2a44;
            color: #fff;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
            flex-shrink: 0;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 15px 17px;
            box-sizing: border-box;
        }}
        .kgrid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 6px 0;
        }}
        .kgrid.k6 {{
            grid-template-columns: repeat(6, minmax(0, 1fr));
        }}
        @media (max-width: 1100px) {{
            .kgrid, .kgrid.k6 {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        .kpi {{
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-top: 4px solid #3d9a84;
            border-radius: 12px;
            padding: 12px 14px;
            box-sizing: border-box;
        }}
        .kpi.y {{
            border-top-color: #e0a526;
        }}
        .kl {{
            font-size: 12px;
            color: #5b6475;
        }}
        .kv {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 21px;
            margin: 2px 0;
            color: #1f2a44;
        }}
        .kv2 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 15px;
            margin: 2px 0;
            color: #1f2a44;
        }}
        .rule {{
            border-top: 1px dashed #d9d3c7;
            margin: 8px 0 6px 0;
        }}
        .chg {{
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 999px;
            white-space: nowrap;
        }}
        .pos {{ background: #d6ece5; color: #14513f; }}
        .neg {{ background: #fbe3e0; color: #8f2a24; }}
        .mute {{ background: #f0ebe2; color: #5b6475; }}
        .neu {{ background: #e4e7ee; color: #2b3550; }}
        .pill {{
            display: inline-block;
            font-size: 11px;
            font-weight: 500;
            padding: 3px 9px;
            border-radius: 999px;
            background: #f0ebe2;
            color: #1f2a44;
            margin: 2px 4px 2px 0;
        }}
        .pill.g {{ background: #d6ece5; color: #14513f; }}
        .pill.y {{ background: #fdf0cf; color: #7a5600; }}
        .pill.r {{ background: #fbe3e0; color: #8f2a24; }}
        .out {{
            border-radius: 10px;
            padding: 9px 11px;
            background: #eaf4f0;
        }}
        .out .t {{
            font-size: 11px;
            color: #5b6475;
        }}
        .out .v {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #1f2a44;
        }}
        table.t {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 6px;
        }}
        table.t th, table.t td {{
            padding: 7px 8px;
            border-bottom: 1px solid #ece7dd;
            text-align: left;
            vertical-align: middle;
        }}
        table.t th {{
            font-size: 11px;
            color: #5b6475;
            font-weight: 600;
            background: #faf7f2;
        }}
        table.t td.r, table.t th.r {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .hm td {{
            text-align: center;
            font-size: 12px;
            font-weight: 500;
            padding: 7px 4px;
            border: 2px solid #fff;
            border-radius: 4px;
        }}
        .hm th {{
            font-size: 11px;
            color: #5b6475;
            padding: 4px;
            text-align: center;
            background: none;
            border: none;
        }}
        ol.lst {{
            margin: 6px 0 0 0;
            padding-left: 18px;
            font-size: 12px;
            line-height: 1.5;
        }}
        ol.lst li {{
            margin: 4px 0;
        }}
        ol.lst small {{
            color: #5b6475;
        }}
        .info {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #e4dfd6;
            color: #1f2a44;
            font-size: 10px;
            font-weight: 700;
            font-style: normal;
            cursor: help;
            position: relative;
            vertical-align: middle;
            margin-left: 4px;
        }}
        .info:hover::after {{
            content: attr(data-tip);
            position: absolute;
            left: 50%;
            bottom: 130%;
            transform: translateX(-50%);
            width: 240px;
            padding: 8px 10px;
            border-radius: 8px;
            background: #1f2a44;
            color: #fff;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 400;
            line-height: 1.4;
            text-align: left;
            white-space: normal;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
            z-index: 9999;
            pointer-events: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def baslik(title: str, sub: str = ""):
    """Sayfa üst başlığını ve global CSS'i basar."""
    eslesme = {
        "Yönetici Özeti": "yonetici",
        "Satış & Fiyat": "satis",
        "Pazarlama & Müşteri": "pazarlama",
        "Tedarik Zinciri": "tedarik",
        "Veri Kalitesi": "kalite",
        "Veri setini yükle": "yukle",
    }
    aktif = eslesme.get(title, "")
    st.session_state["_aktif_sayfa"] = aktif
    _inject_css(aktif)
    if aktif == "yukle":
        st.markdown(
            "<style>section[data-testid='stSidebar']{display:none !important;}</style>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='rr-page-header'><h1>{title}</h1><p>{sub}</p></div>",
        unsafe_allow_html=True,
    )


def bolum(num: int, title: str, sub: str = ""):
    """Numaralı bölüm başlığını basar."""
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='h'><span class='num'>{num}</span><span>{title}</span></div>{sub_html}",
        unsafe_allow_html=True,
    )


def _preset_dates(preset: str, min_d: datetime.date, max_d: datetime.date, ref_a: datetime.date = None, ref_b: datetime.date = None):
    y = max_d.year
    if preset == "Geçtiğimiz yıl":
        a, b = datetime.date(y, 1, 1), datetime.date(y, 12, 31)
    elif preset == "Geçtiğimiz ay":
        a, b = datetime.date(y, 12, 1), datetime.date(y, 12, 31)
    elif preset == "Son çeyrek":
        a, b = datetime.date(y, 10, 1), datetime.date(y, 12, 31)
    elif preset == "İlk yarıyıl":
        a, b = datetime.date(y, 1, 1), datetime.date(y, 6, 30)
    elif preset == "Son yarıyıl":
        a, b = datetime.date(y, 7, 1), datetime.date(y, 12, 31)
    elif preset == "Önceki dönem" and ref_a is not None and ref_b is not None:
        uzunluk = max(1, (ref_b - ref_a).days + 1)
        b = max(min_d, ref_a - datetime.timedelta(days=1))
        a = max(min_d, b - datetime.timedelta(days=uzunluk - 1))
    else:
        return None, None
    return max(min_d, min(max_d, a)), max(min_d, min(max_d, b))


def _sayfa_degistir(hedef_id: str):
    sayfa_map = st.session_state.get("_sayfa_nesneleri") or {}
    if hedef_id in sayfa_map:
        st.switch_page(sayfa_map[hedef_id])


def filtreler(tum: pd.DataFrame) -> tuple[pd.DataFrame, datetime.date, datetime.date, datetime.date, datetime.date]:
    """RafRadar_dashboard (3).html'deki sol menüyü (Logo, ROL, SAYFALAR, FİLTRELER, İlk açılış ekranı) birebir oluşturur."""
    min_d = tum["gun"].min().date()
    max_d = tum["gun"].max().date()

    def_a = max(min_d, min(max_d, datetime.date(2020, 11, 1)))
    def_b = max(min_d, min(max_d, datetime.date(2020, 12, 14)))
    def_ca = max(min_d, min(max_d, datetime.date(2020, 9, 1)))
    def_cb = max(min_d, min(max_d, datetime.date(2020, 10, 31)))

    ss = st.session_state
    ss.setdefault("a", def_a)
    ss.setdefault("b", def_b)
    ss.setdefault("ca", def_ca)
    ss.setdefault("cb", def_cb)
    ss.setdefault("f_kat", "Tümü")
    ss.setdefault("f_urun", "Tümü")
    ss.setdefault("f_sehir", "Tümü")

    aktif_sayfa = ss.get("_aktif_sayfa", "yonetici")

    with st.sidebar:
        # 1. LOGO & BAŞLIK (HTML ile birebir)
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <svg width="28" height="26" viewBox="0 0 28 26" aria-hidden="true">
                    <rect x="2" y="10" width="6" height="14" fill="#1f2a44"/>
                    <rect x="11" y="3" width="6" height="21" fill="#3d9a84"/>
                    <rect x="20" y="14" width="6" height="10" fill="#e0a526"/>
                </svg>
                <span style="font-family:Poppins,sans-serif;font-weight:700;font-size:20px;color:#1f2a44">RafRadar</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. ROL SEÇİCİ (HTML'deki gibi: Yönetici, Satış, Pazarlama, Tedarik Zinciri)
        st.markdown("<div class='rr-side-hdr'>ROL</div>", unsafe_allow_html=True)
        rol_ids = [r[0] for r in ROLLER_META]
        rol_adlar = [r[1] for r in ROLLER_META]
        mevcut_rol_idx = rol_ids.index(aktif_sayfa) if aktif_sayfa in rol_ids else 0
        secilen_rol_ad = st.selectbox(
            "ROL",
            rol_adlar,
            index=mevcut_rol_idx,
            key=f"sb_rol_{aktif_sayfa}",
            label_visibility="collapsed",
        )
        hedef_rol_id = rol_ids[rol_adlar.index(secilen_rol_ad)]
        if hedef_rol_id != aktif_sayfa and aktif_sayfa in rol_ids:
            _sayfa_degistir(hedef_rol_id)

        # 3. SAYFALAR MENÜSÜ (HTML'deki renkli noktalı 5 sayfa)
        st.markdown("<div class='rr-side-hdr' style='margin-top:12px'>SAYFALAR</div>", unsafe_allow_html=True)
        for pid, pname, pcolor, picon in SAYFALAR_META:
            aktif_mi = pid == aktif_sayfa
            if st.button(
                f"{picon}  {pname}",
                key=f"sb_nav_{pid}",
                type="primary" if aktif_mi else "secondary",
                use_container_width=True,
            ):
                if not aktif_mi:
                    _sayfa_degistir(pid)

        # 4. FİLTRELER BÖLÜMÜ (HTML ile birebir sıra ve yapı)
        st.markdown("<div class='rr-side-hdr' style='margin-top:14px'>FİLTRELER</div>", unsafe_allow_html=True)

        # Tarih aralığı
        hazir_secenekler = ["Geçtiğimiz yıl", "Geçtiğimiz ay", "Son çeyrek", "İlk yarıyıl", "Son yarıyıl", "Özel (takvim)"]
        hazir_varsayilan = ss.get("f_hazir", "Özel (takvim)")
        hazir_idx = hazir_secenekler.index(hazir_varsayilan) if hazir_varsayilan in hazir_secenekler else 5
        hazir = st.selectbox("Tarih aralığı", hazir_secenekler, index=hazir_idx, key="f_hazir")
        if hazir != "Özel (takvim)":
            pa, pb = _preset_dates(hazir, min_d, max_d)
            if pa and pb:
                ss["a"], ss["b"] = pa, pb
            a, b = ss["a"], ss["b"]
            st.markdown(f"<div style='margin:-6px 0 6px 2px;font-size:11px;color:#5b6475'>{tarih_etiketi(a, b)}</div>", unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            a = c1.date_input("Başlangıç", value=ss["a"], min_value=min_d, max_value=max_d, format="DD.MM.YYYY", key="inp_a", label_visibility="collapsed")
            b = c2.date_input("Bitiş", value=ss["b"], min_value=min_d, max_value=max_d, format="DD.MM.YYYY", key="inp_b", label_visibility="collapsed")
            if a > b:
                a, b = b, a
            ss["a"], ss["b"] = a, b

        # Karşılaştırma aralığı
        kars_secenekler = ["Geçtiğimiz yıl", "Geçtiğimiz ay", "Son çeyrek", "İlk yarıyıl", "Son yarıyıl", "Önceki dönem", "Özel (takvim)"]
        kars_varsayilan = ss.get("f_kars", "Özel (takvim)")
        kars_idx = kars_secenekler.index(kars_varsayilan) if kars_varsayilan in kars_secenekler else 6
        kars = st.selectbox("Karşılaştırma aralığı", kars_secenekler, index=kars_idx, key="f_kars")
        if kars != "Özel (takvim)":
            pca, pcb = _preset_dates(kars, min_d, max_d, a, b)
            if pca and pcb:
                ss["ca"], ss["cb"] = pca, pcb
            ca, cb = ss["ca"], ss["cb"]
            st.markdown(f"<div style='margin:-6px 0 6px 2px;font-size:11px;color:#5b6475'>{tarih_etiketi(ca, cb)}</div>", unsafe_allow_html=True)
        else:
            cc1, cc2 = st.columns(2)
            ca = cc1.date_input("Karş. başlangıç", value=ss["ca"], min_value=min_d, max_value=max_d, format="DD.MM.YYYY", key="inp_ca", label_visibility="collapsed")
            cb = cc2.date_input("Karş. bitiş", value=ss["cb"], min_value=min_d, max_value=max_d, format="DD.MM.YYYY", key="inp_cb", label_visibility="collapsed")
            if ca > cb:
                ca, cb = cb, ca
            ss["ca"], ss["cb"] = ca, cb

        # Kategori, Ürün, Şehir
        kategoriler = ["Tümü"] + sorted(tum["Category"].dropna().unique().tolist())
        f_kat = st.selectbox("Kategori", kategoriler, index=kategoriler.index(ss["f_kat"]) if ss["f_kat"] in kategoriler else 0, key="f_kat")

        df_kat = tum if f_kat == "Tümü" else tum[tum["Category"] == f_kat]
        en_cok_urunler = df_kat.groupby("Description")["net"].sum().nlargest(250).index.tolist()
        urun_secenekler = ["Tümü"] + en_cok_urunler
        f_urun = st.selectbox("Ürün", urun_secenekler, index=urun_secenekler.index(ss["f_urun"]) if ss["f_urun"] in urun_secenekler else 0, key="f_urun")

        sehirler = ["Tümü"] + sorted(tum["City"].dropna().unique().tolist())
        f_sehir = st.selectbox("Şehir", sehirler, index=sehirler.index(ss["f_sehir"]) if ss["f_sehir"] in sehirler else 0, key="f_sehir")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Filtreleri sıfırla", key="sb_reset_btn", use_container_width=True):
            for k in ("a", "b", "ca", "cb", "f_kat", "f_urun", "f_sehir", "f_hazir", "f_kars", "inp_a", "inp_b", "inp_ca", "inp_cb"):
                ss.pop(k, None)
            st.rerun()

        st.markdown("<div style='border-top:1px dashed #cfc9bd;margin:10px 0 6px'></div>", unsafe_allow_html=True)
        if st.button("📂 İlk açılış ekranı (veri yükle)", key="sb_yukle_btn", use_container_width=True):
            _sayfa_degistir("yukle")

    m = tum
    if f_kat != "Tümü":
        m = m[m["Category"] == f_kat]
    if f_urun != "Tümü":
        m = m[m["Description"] == f_urun]
    if f_sehir != "Tümü":
        m = m[m["City"] == f_sehir]

    return m, a, b, ca, cb


def plot(fig):
    """Plotly grafiğini RafRadar temasıyla basar."""
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", color="#1f2a44", size=11),
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="#1f2a44", font_color="white", font_family="Inter, sans-serif", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def xlsx(df: pd.DataFrame, sheet_name: str = "Veri") -> bytes:
    """DataFrame'i Excel (.xlsx) byte dizisine dönüştürür."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()
