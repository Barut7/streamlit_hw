"""RafRadar ortak widget, tablo, KPI kartı ve önbellekli sarmalayıcılar."""
from __future__ import annotations

import streamlit as st
from common import analytics as an
from common import ui

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
AY_AD = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]


def fk() -> tuple:
    """Aktif filtre önbellek anahtarı."""
    ss = st.session_state
    return (
        ss.get("dosya_adi", ""),
        ss.get("f_kat", "Tümü"),
        ss.get("f_urun", "Tümü"),
        ss.get("f_sehir", "Tümü"),
    )


@st.cache_data(show_spinner=False)
def ozet_c(_m, K, a, b) -> dict:
    return an.ozet(_m, a, b)


@st.cache_data(show_spinner="Churn modeli eğitiliyor…")
def churn_c(_tum, vid):
    return an.churn_modeli(_tum)


@st.cache_data(show_spinner=False)
def birlikte_c(_m, K, a, b):
    x = an.araliga_kes(_m, a, b)
    return an.birlikte(x, limit=12)


def rel(a, b) -> float:
    return (a / b - 1.0) if b else 0.0


def chg(x: float, vurgu: bool = True) -> str:
    sinif = "mute" if not vurgu else ("neg" if x < 0 else "pos")
    ok = "▲ %" if x >= 0 else "▼ %"
    return f"<span class='chg {sinif}'>{ok}{ui.sayi(abs(x * 100), 1)}</span>"


def chg_p(d: float, vurgu: bool = True) -> str:
    sinif = "neu" if vurgu else "mute"
    ok = "▲ " if d >= 0 else "▼ "
    return f"<span class='chg {sinif}'>{ok}{ui.sayi(abs(d * 100), 1)} puan</span>"


def ust_bilgi(S: dict, C: dict, a, b, ca, cb) -> str:
    ayni = S["gun"] == C["gun"]
    et = ui.tarih_etiketi
    not_txt = (
        "Aralıklar eşit uzunlukta: vurgulu fark nominal değerlere göre"
        if ayni
        else "Aralık uzunlukları farklı: vurgulu fark günlük ortalamaya göre"
    )
    return (
        f"<div style='margin:4px 0 6px'>"
        f"<span class='pill g'>Seçili: {et(a, b)} · {S['gun']} gün</span>"
        f"<span style='font-size:12px;color:#5b6475;margin:0 4px'>karşılaştırma</span>"
        f"<span class='pill'>{et(ca, cb)} · {C['gun']} gün</span>"
        f"<span class='pill y'>{not_txt}</span>"
        f"</div>"
    )


def kart(baslik: str, nom: str, nom_c: str, gun_et: str, gun_v: str, gun_c: str, renk: str = "#3d9a84") -> str:
    return (
        f"<div class='kpi' style='border-top-color:{renk}'>"
        f"<div class='kl'>{baslik}</div>"
        f"<div class='kv'>{nom}</div>{nom_c}"
        f"<div class='rule'></div>"
        f"<div class='kl'>{gun_et}</div>"
        f"<div class='kv2'>{gun_v}</div>{gun_c}"
        f"</div>"
    )


def kgrid(cards: list[str], cls: str = ""):
    st.markdown(f"<div class='kgrid {cls}'>{''.join(cards)}</div>", unsafe_allow_html=True)


def not_kutusu(txt: str):
    st.markdown(f"<p class='sub' style='margin:6px 0 12px'>{txt}</p>", unsafe_allow_html=True)


def vurgu(txt: str):
    st.markdown(
        f"<div style='font-size:13px;font-weight:600;color:#1f2a44;margin:4px 0 8px'>{txt}</div>",
        unsafe_allow_html=True,
    )


def info(tip_txt: str) -> str:
    temiz = tip_txt.replace("'", "&#39;").replace('"', "&quot;")
    return f"<span class='info' data-tip='{temiz}'>i</span>"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return f"rgb({r},{g},{b})"


def tablo(basliklar: list[str], satirlar: list[list[str]], sag: tuple = (), stil: list[str] | None = None) -> str:
    th = "".join(f"<th class='{'r' if i in sag else ''}'>{b}</th>" for i, b in enumerate(basliklar))
    trs = []
    for idx, s in enumerate(satirlar):
        row_style = f" style='{stil[idx]}'" if stil and idx < len(stil) and stil[idx] else ""
        tds = "".join(f"<td class='{'r' if i in sag else ''}'>{c}</td>" for i, c in enumerate(s))
        trs.append(f"<tr{row_style}>{tds}</tr>")
    return f"<table class='t'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def sapma_tablo(rows: list[str], cols: list[str], mat: list[list[int]]) -> str:
    """100 = ortalama endeks ısı haritası tablosu."""
    th = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for r_lbl, r_vals in zip(rows, mat):
        tds = [f"<td style='text-align:left;font-weight:600;background:#faf7f2;color:#1f2a44'>{r_lbl}</td>"]
        for v in r_vals:
            if v >= 100:
                t = min(1.0, (v - 100) / 80.0)
                bg = mix("#f4f1ea", "#2f8a73", t)
                fg = "#fff" if t > 0.55 else "#1f2a44"
            else:
                t = min(1.0, (100 - v) / 70.0)
                bg = mix("#f4f1ea", "#d9622b", t * 0.65)
                fg = "#fff" if t > 0.7 else "#1f2a44"
            tds.append(f"<td style='background:{bg};color:{fg};font-weight:600'>{int(round(v))}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    return f"<div style='overflow-x:auto'><table class='t hm'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
