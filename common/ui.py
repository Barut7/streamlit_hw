"""RafRadar ortak arayüz (UI), biçimlendirme, tema CSS ve filtre bileşenleri."""
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


def _inject_css():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        :root {
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
        }
        .stApp {
            background-color: var(--bg) !important;
            color: var(--fg);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 4.5rem !important;
            max-width: 1440px !important;
        }
        section[data-testid="stSidebar"] {
            background-color: var(--side) !important;
            border-right: 1px solid var(--line) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #e4dfd6 !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 2px rgba(31, 42, 68, 0.02);
        }
        h1, h2, h3, .h {
            font-family: 'Poppins', sans-serif !important;
            color: #1f2a44;
        }
        .rr-page-header {
            margin-bottom: 10px;
        }
        .rr-page-header h1 {
            font-size: 27px !important;
            font-weight: 700 !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            color: #1f2a44 !important;
        }
        .rr-page-header p {
            margin: 4px 0 0 0 !important;
            font-size: 14px !important;
            font-style: italic;
            color: #5b6475 !important;
        }
        .h {
            font-weight: 600;
            font-size: 15px;
            margin: 0;
            display: flex;
            align-items: center;
            color: #1f2a44;
        }
        .sub {
            font-size: 12px;
            color: #5b6475;
            margin: 2px 0 6px 0;
            line-height: 1.4;
        }
        .num {
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
        }
        .card {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-radius: 12px;
            padding: 15px 17px;
            box-sizing: border-box;
        }
        .kgrid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 6px 0;
        }
        .kgrid.k6 {
            grid-template-columns: repeat(6, minmax(0, 1fr));
        }
        @media (max-width: 1100px) {
            .kgrid, .kgrid.k6 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .kpi {
            background: #ffffff;
            border: 1px solid #e4dfd6;
            border-top: 4px solid #3d9a84;
            border-radius: 12px;
            padding: 12px 14px;
            box-sizing: border-box;
        }
        .kpi.y {
            border-top-color: #e0a526;
        }
        .kl {
            font-size: 12px;
            color: #5b6475;
        }
        .kv {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 21px;
            margin: 2px 0;
            color: #1f2a44;
        }
        .kv2 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 15px;
            margin: 2px 0;
            color: #1f2a44;
        }
        .rule {
            border-top: 1px dashed #d9d3c7;
            margin: 8px 0 6px 0;
        }
        .chg {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .pos { background: #d6ece5; color: #14513f; }
        .neg { background: #fbe3e0; color: #8f2a24; }
        .mute { background: #f0ebe2; color: #5b6475; }
        .neu { background: #e4e7ee; color: #2b3550; }
        .pill {
            display: inline-block;
            font-size: 11px;
            font-weight: 500;
            padding: 3px 9px;
            border-radius: 999px;
            background: #f0ebe2;
            color: #1f2a44;
            margin: 2px 4px 2px 0;
        }
        .pill.g { background: #d6ece5; color: #14513f; }
        .pill.y { background: #fdf0cf; color: #7a5600; }
        .pill.r { background: #fbe3e0; color: #8f2a24; }
        .out {
            border-radius: 10px;
            padding: 9px 11px;
            background: #eaf4f0;
        }
        .out .t {
            font-size: 11px;
            color: #5b6475;
        }
        .out .v {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #1f2a44;
        }
        table.t {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 6px;
        }
        table.t th, table.t td {
            padding: 7px 8px;
            border-bottom: 1px solid #ece7dd;
            text-align: left;
            vertical-align: middle;
        }
        table.t th {
            font-size: 11px;
            color: #5b6475;
            font-weight: 600;
            background: #faf7f2;
        }
        table.t td.r, table.t th.r {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        .hm td {
            text-align: center;
            font-size: 12px;
            font-weight: 500;
            padding: 7px 4px;
            border: 2px solid #fff;
            border-radius: 4px;
        }
        .hm th {
            font-size: 11px;
            color: #5b6475;
            padding: 4px;
            text-align: center;
            background: none;
            border: none;
        }
        ol.lst {
            margin: 6px 0 0 0;
            padding-left: 18px;
            font-size: 12px;
            line-height: 1.5;
        }
        ol.lst li {
            margin: 4px 0;
        }
        ol.lst small {
            color: #5b6475;
        }
        .info {
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
        }
        .info:hover::after {
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def baslik(title: str, sub: str = ""):
    """Sayfa üst başlığını ve global CSS'i basar."""
    _inject_css()
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
        m = max_d.month - 1 or 12
        yy = y if max_d.month > 1 else y - 1
        a = datetime.date(yy, m, 1)
        last_day = (datetime.date(yy + (m // 12), (m % 12) + 1, 1) - datetime.timedelta(days=1)).day
        b = datetime.date(yy, m, last_day)
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


def filtreler(tum: pd.DataFrame) -> tuple[pd.DataFrame, datetime.date, datetime.date, datetime.date, datetime.date]:
    """Sol yan menüde tarih, karşılaştırma ve kırılım filtrelerini oluşturur."""
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

    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:9px;padding-bottom:10px;border-bottom:1px solid #e4dfd6;margin-bottom:10px">
                <svg width="28" height="26" viewBox="0 0 28 26">
                    <rect x="2" y="10" width="6" height="14" fill="#1f2a44"/>
                    <rect x="11" y="3" width="6" height="21" fill="#3d9a84"/>
                    <rect x="20" y="14" width="6" height="10" fill="#e0a526"/>
                </svg>
                <div>
                    <div style="font-family:Poppins,sans-serif;font-weight:700;font-size:19px;line-height:1.1;color:#1f2a44">RafRadar</div>
                    <div style="font-size:11px;color:#5b6475">Kitap &amp; Kırtasiye B2B</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='font-size:11px;font-weight:700;color:#5b6475;letter-spacing:.04em;margin-bottom:4px'>FİLTRELER</div>", unsafe_allow_html=True)

        hazir_secenekler = ["Özel (takvim)", "Geçtiğimiz yıl", "Geçtiğimiz ay", "Son çeyrek", "İlk yarıyıl", "Son yarıyıl"]
        hazir = st.selectbox("Hazır aralık", hazir_secenekler, index=0, key="f_hazir")
        if hazir != "Özel (takvim)":
            pa, pb = _preset_dates(hazir, min_d, max_d)
            if pa and pb:
                ss["a"], ss["b"] = pa, pb

        c1, c2 = st.columns(2)
        a = c1.date_input("Başlangıç", value=ss["a"], min_value=min_d, max_value=max_d, key="inp_a")
        b = c2.date_input("Bitiş", value=ss["b"], min_value=min_d, max_value=max_d, key="inp_b")
        if a > b:
            a, b = b, a
        ss["a"], ss["b"] = a, b

        kars_secenekler = ["Özel (takvim)", "Önceki dönem", "Geçtiğimiz yıl", "Geçtiğimiz ay", "Son çeyrek", "İlk yarıyıl", "Son yarıyıl"]
        kars = st.selectbox("Karşılaştırma aralığı", kars_secenekler, index=0, key="f_kars")
        if kars != "Özel (takvim)":
            pca, pcb = _preset_dates(kars, min_d, max_d, a, b)
            if pca and pcb:
                ss["ca"], ss["cb"] = pca, pcb

        cc1, cc2 = st.columns(2)
        ca = cc1.date_input("Karş. başlangıç", value=ss["ca"], min_value=min_d, max_value=max_d, key="inp_ca")
        cb = cc2.date_input("Karş. bitiş", value=ss["cb"], min_value=min_d, max_value=max_d, key="inp_cb")
        if ca > cb:
            ca, cb = cb, ca
        ss["ca"], ss["cb"] = ca, cb

        kategoriler = ["Tümü"] + sorted(tum["Category"].dropna().unique().tolist())
        f_kat = st.selectbox("Kategori", kategoriler, index=kategoriler.index(ss["f_kat"]) if ss["f_kat"] in kategoriler else 0, key="f_kat")

        df_kat = tum if f_kat == "Tümü" else tum[tum["Category"] == f_kat]
        en_cok_urunler = df_kat.groupby("Description")["net"].sum().nlargest(250).index.tolist()
        urun_secenekler = ["Tümü"] + en_cok_urunler
        f_urun = st.selectbox("Ürün", urun_secenekler, index=urun_secenekler.index(ss["f_urun"]) if ss["f_urun"] in urun_secenekler else 0, key="f_urun")

        sehirler = ["Tümü"] + sorted(tum["City"].dropna().unique().tolist())
        f_sehir = st.selectbox("Şehir", sehirler, index=sehirler.index(ss["f_sehir"]) if ss["f_sehir"] in sehirler else 0, key="f_sehir")

        if st.button("Filtreleri sıfırla", use_container_width=True):
            for k in ("a", "b", "ca", "cb", "f_kat", "f_urun", "f_sehir", "f_hazir", "f_kars", "inp_a", "inp_b", "inp_ca", "inp_cb"):
                ss.pop(k, None)
            st.rerun()

        dosya_adi = ss.get("dosya_adi", "B2B_Transaction_Data_")
        st.markdown(
            f"<div style='margin-top:8px;padding:7px 9px;border-radius:8px;background:#eaf4f0;font-size:11px;color:#14513f'>"
            f"<b>✓ Veri:</b> {dosya_adi}<br>{sayi(len(tum))} satır · {min_d.year}</div>",
            unsafe_allow_html=True,
        )

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
