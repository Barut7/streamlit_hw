"""RafRadar - tek dosya. Önce veri yüklenir, sonra sayfalar açılır.
Altta common/ klasörü (hesaplama ve ortak arayüz) ayrı kalır; sayfalar bu dosyadadır."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import analytics as an
from common import analytics2 as an2
from common import chat, data, ui
from common import widgets as w

st.set_page_config(page_title="RafRadar", page_icon="📚", layout="wide")
KOK = Path(__file__).resolve().parent


# ======================================================================
# SAYFA: Veri yükle
# ======================================================================
def sayfa_yukle():
    ui.baslik("Veri setini yükle", "Dashboard, yüklediğin veriden hesaplanır. Hazır örnek dosya seçili; istersen kendi dosyanı yükleyebilirsin.")

    hazir_mi = "df" in st.session_state
    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 14px'>"
        f"<span class='pill {'g' if hazir_mi else ''}' style='{'background:#1f2a44;color:#fff' if not hazir_mi else ''}'>{'✓ ' if hazir_mi else ''}1 · Dosya seç</span>"
        f"<span class='pill {'g' if hazir_mi else ''}'>{'✓ ' if hazir_mi else ''}2 · Kontrol</span>"
        f"<span class='pill {'g' if hazir_mi else ''}'>{'✓ ' if hazir_mi else ''}3 · İşleme</span>"
        f"<span class='pill' style='{'background:#1f2a44;color:#fff' if hazir_mi else ''}'>{'✓ ' if hazir_mi else ''}4 · Hazır</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    ORNEK = data.ornek_dosyalar()
    sec_ornek = "Örnek veri (hazır dosya)"
    sec_kendi = "Kendi dosyamı yükle"
    secenekler = ([sec_ornek] if ORNEK else []) + [sec_kendi]

    sol, sag = st.columns([1.2, 1])
    dosya_adi = icerik = None
    with sol:
        kaynak = st.radio("Veri kaynağı", secenekler, index=0, horizontal=True, label_visibility="collapsed")
        if kaynak == sec_ornek:
            f = ORNEK[0]
            st.markdown(f"<div class='card' style='border:2px solid #3d9a84;background:#f4faf8;border-radius:14px;padding:16px 18px;margin-bottom:10px'>"
                        f"<div style='font-size:12px;color:#14513f;font-weight:700'>✓ SEÇİLİ DOSYA</div>"
                        f"<div style='font-size:16px;font-weight:700;margin:4px 0 2px'>📄 {f.name}</div>"
                        f"<div style='font-size:12px;color:#5b6475'>{f.stat().st_size / 1048576:.1f} MB · 2020 yılı B2B işlem verisi · 9 zorunlu sütun hazır</div></div>",
                        unsafe_allow_html=True)
            if st.button("Bu dosyayla devam et", type="primary"):
                dosya_adi, icerik = f.name, f.read_bytes()
        else:
            dosya = st.file_uploader("Dosyayı buraya sürükle ya da seç", type=["xlsx", "csv", "parquet"],
                                     help="Excel (.xlsx), CSV veya Parquet · en fazla 200 MB")
            if dosya is not None:
                dosya_adi, icerik = dosya.name, dosya.getvalue()

        st.markdown(
            "<div class='card' style='margin-top:12px;background:#faf7f2'>"
            "<div style='font-weight:600;font-size:13px;margin-bottom:4px'>Dosya yanlışsa ne görürsün?</div>"
            "<div class='sub'>Eksik sütun veya bozuk tarih olduğunda yükleme durur ve neyin eksik olduğu açıkça yazılır:</div>"
            "<div style='margin-top:6px;padding:8px 10px;border-radius:8px;background:#fbe3e0;color:#8f2a24;font-size:11.5px'>"
            "<b>Örnek hata 1:</b> Eksik sütun: <code>NetPrice</code>, <code>Category</code>. Dosyada 9 zorunlu sütunun tamamı olmalı.<br>"
            "<b>Örnek hata 2:</b> <code>InvoiceDate</code> sütunundaki değerler tarih olarak okunamadı."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with sag:
        st.markdown("**Dosyada olması gerekenler**")
        st.dataframe({
            "Sütun adı": data.ZORUNLU,
            "Ne için": ["siparişi tanımlar", "segment ve churn", "tarih filtreleri",
                        "ürün analizleri", "kategori kırılımı", "il bazlı panel",
                        "satış ve stok", "brüt ciro", "indirim ve net ciro"],
        }, hide_index=True, use_container_width=True)

    if icerik is not None:
        with st.status("Veri işleniyor…", expanded=True) as durum:
            st.write("Dosya okunuyor ve kontrol ediliyor")
            df, atilan, hatalar, kalite = data.hazirla(dosya_adi, icerik)
            if hatalar:
                durum.update(label="Dosya uygun değil", state="error")
                for h in hatalar:
                    st.error(h)
            else:
                st.write("Temizlendi, brüt/net/indirim hesaplandı")
                durum.update(label="Veri hazır", state="complete", expanded=False)
                for k in ("a", "b", "ca", "cb"):
                    st.session_state.pop(k, None)
                st.session_state["df"] = df
                st.session_state["atilan"] = atilan
                st.session_state["kalite"] = kalite
                st.session_state["dosya_adi"] = dosya_adi
                st.rerun()

    if "df" in st.session_state:
        d = st.session_state["df"]
        st.success("Veri hazır")
        c = st.columns(5)
        c[0].metric("Kullanılan satır", ui.sayi(len(d)))
        c[1].metric("Müşteri", ui.sayi(d["CustomerID"].nunique()))
        c[2].metric("Ürün", ui.sayi(d["Description"].nunique()))
        c[3].metric("İlk tarih", f'{d["InvoiceDate"].min():%d.%m.%Y}')
        c[4].metric("Son tarih", f'{d["InvoiceDate"].max():%d.%m.%Y}')
        st.caption(f'{ui.sayi(st.session_state["atilan"])} satır temizlikte çıkarıldı.')
        if st.button("Dashboard'a geç →", type="primary"):
            st.switch_page(yonetici)


# ======================================================================
# SAYFA: Yönetici Özeti
# ======================================================================
def sayfa_yonetici():
    @st.cache_data(show_spinner="Tahmin hesaplanıyor…")
    def c_baz(_m, K):
        return an2.baz_tahmin30(_m)


    @st.cache_data(show_spinner=False)
    def c_norm(_tum, _m, K):
        return an2.norm_asimi(_tum, _m)


    @st.cache_data(show_spinner=False)
    def c_abc(_m, K):
        q, kum, abc, xyz = an2.abc_xyz(_m)
        return int(((abc == "A") & (xyz == "Z")).sum()), int((abc == "A").sum())


    @st.cache_data(show_spinner=False)
    def c_yavas(_m, K, bitis):
        return an2.yavas_urunler(_m, bitis)


    tum = data.veri()
    vid = (st.session_state.get("dosya_adi", ""), len(tum))
    ui.baslik("Yönetici Özeti", "Üç departmanın seçili dönemdeki durumu ve önümüzdeki 30 günde ne yapmalıyız?")
    m, a, b, ca, cb = ui.filtreler(tum)
    K = w.fk()
    if m.empty:
        st.warning("Bu filtrelerle veri yok. Filtreleri gevşetip tekrar dene.")
        st.stop()
    et = ui.tarih_etiketi
    bit = tum["gun"].max()
    S, C = w.ozet_c(m, K, a, b), w.ozet_c(m, K, ca, cb)
    hN, hD = S["gun"] == C["gun"], S["gun"] != C["gun"]
    ch = an2.musteri_degisim(tum, m, a, b, ca, cb)
    perC = S["net"] / ch["aS"] if ch["aS"] else 0
    perCc = C["net"] / ch["aC"] if ch["aC"] else 0
    dpc = S["dnet"] / S["dcust"] if S["dcust"] else 0
    dpcc = C["dnet"] / C["dcust"] if C["dcust"] else 0
    st.markdown(w.ust_bilgi(S, C, a, b, ca, cb), unsafe_allow_html=True)


    st.markdown("<div style='display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin:8px 0 2px'><span><span style='color:#3d9a84'>●</span> <b>Satış & Fiyat</b></span>"
                "<span><span style='color:#d9622b'>●</span> <b>Pazarlama & Müşteri</b></span><span><span style='color:#3b6fd1'>●</span> <b>Tedarik Zinciri</b></span></div>", unsafe_allow_html=True)
    w.kgrid([
        w.kart("Net ciro", ui.sayi(S["net"] / 1e6, 2) + " M ₺", w.chg(w.rel(S["net"], C["net"]), hN), "Günlük ort.", ui.sayi(S["dnet"] / 1000, 1) + " bin ₺", w.chg(w.rel(S["dnet"], C["dnet"]), hD)),
        w.kart("Sipariş adedi", ui.sayi(S["ord"]), w.chg(w.rel(S["ord"], C["ord"]), hN), "Günlük ort.", ui.sayi(S["dord"], 1), w.chg(w.rel(S["dord"], C["dord"]), hD)),
        w.kart("İndirim oranı", "%" + ui.sayi(S["disc"] * 100, 1), w.chg_p(S["disc"] - C["disc"], hN), "Günlük ort.", "%" + ui.sayi(S["dd"] * 100, 1), w.chg_p(S["dd"] - C["dd"], hD), "#e0a526"),
        w.kart("Aktif müşteri", ui.sayi(ch["aS"]), w.chg(w.rel(ch["aS"], ch["aC"]), hN), "Günlük ort.", ui.sayi(S["dcust"]), w.chg(w.rel(S["dcust"], C["dcust"]), hD), "#d9622b"),
        w.kart("Müşteri başı ciro", ui.sayi(perC / 1000, 1) + " bin ₺", w.chg(w.rel(perC, perCc), hN), "Günlük ort.", ui.sayi(dpc) + " ₺", w.chg(w.rel(dpc, dpcc), hD), "#d9622b"),
        w.kart("Satılan adet", ui.sayi(S["q"] / 1000, 1) + " bin", w.chg(w.rel(S["q"], C["q"]), hN), "Günlük ort.", ui.sayi(S["dq"]) + " adet", w.chg(w.rel(S["dq"], C["dq"]), hD), "#3b6fd1"),
    ], "k6")
    w.not_kutusu("Aralıklar eşit uzunlukta olursa vurgu nominal değerlere, farklıysa günlük ortalamalara geçer. 1, 2, 4 ve 5. bölümler seçtiğin tarihten bağımsızdır; 3. bölüm seçtiğin tarihe göre değişir. Kategori, ürün ve şehir filtreleri hepsini etkiler.")

    # ---------- 1-2 ----------
    p1, p2 = st.columns([1.9, 1])
    yillar = sorted(set(tum["gun"].dt.year.unique().tolist()) | {tum["gun"].dt.year.min() - 1, tum["gun"].dt.year.max() + 1})
    with p1, st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            ui.bolum(1, "Yıllık genel bakış", "Tarih filtresinden bağımsız · çubuk: aylık net ciro · çizgi: indirim oranı")
        with h2:
            yil = st.selectbox("Yıl", yillar, index=yillar.index(bit.year), key="yon_yil")
        ay = an2.aylik(m, yil)
        if ay is None:
            st.markdown(f"<div style='padding:60px 16px;text-align:center;background:#faf7f2;border-radius:10px;font-size:13px;color:#5b6475'><b style='color:#1f2a44'>{yil} için veri yok.</b><br>Veri seti bu yılı kapsamıyor.</div>", unsafe_allow_html=True)
        else:
            mxm = ay["net"].max()
            pk = int(ay["net"].values.argmax())
            vl = int(ay.loc[ay["net"] > 0, "net"].values.argmin()) if (ay["net"] > 0).any() else 0
            renk = ["#1d6b58" if i == pk else ("#9fcfc2" if i == vl else "#3d9a84") for i in range(12)]
            fig = go.Figure()
            fig.add_bar(x=ui.AYLAR, y=ay["net"] / 1e6, marker_color=renk, name="Net ciro", customdata=np.c_[ay["net"] / 1e6, ay["disc"] * 100],
                        hovertemplate="%{x}<br>Net ciro %{customdata[0]:.2f} M ₺<br>İndirim oranı %%{customdata[1]:.1f}<extra></extra>")
            fig.add_scatter(x=ui.AYLAR, y=ay["disc"] * 100, yaxis="y2", mode="lines+markers", line=dict(color="#e0a526", width=2), name="İndirim %",
                            hovertemplate="%{x}<br>İndirim %{y:.1f}%<extra></extra>")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", showlegend=False,
                              yaxis=dict(title="M ₺", rangemode="tozero"),
                              yaxis2=dict(overlaying="y", side="right", range=[35, 44], showgrid=False, ticksuffix="%"))
            fig.add_annotation(x=ui.AYLAR[pk], y=mxm / 1e6, text=f"zirve {ui.sayi(mxm / 1e6, 2)}", showarrow=False, yshift=12, font=dict(size=11))
            ui.plot(fig)
            tot = ay["net"].sum()
            gross = ay["gross"].sum()
            st.markdown(f"<span class='pill g'>Yıllık net ciro {ui.sayi(tot / 1e6, 2)} M ₺</span><span class='pill y'>İndirim oranı %{ui.sayi((1 - tot / gross) * 100, 1)}</span>"
                        "<span class='pill'>Aya gel: ciro ve indirim oranı görünür</span>", unsafe_allow_html=True)

    pm = an.pareto(m, "CustomerID")
    pu = an.pareto(m, "Description")
    with p2, st.container(border=True):
        ui.bolum(2, "Portföy yapısı", "Müşteri ve ürünler ciro payına göre A (%80) · B (%15) · C (%5)")

        def seritler(p, ad, birim):
            n = len(p)
            c = {k: int((p["seg"] == k).sum()) for k in "ABC"}
            renk = {"A": "#3d9a84", "B": "#e0a526", "C": "#d9622b"}
            bar = "".join(f"<div style='flex:{max(c[k], 0.001)} 1 0;background:{renk[k]};color:#fff;font-size:12px;font-weight:700;text-align:center;padding:5px 0'>{k}</div>" for k in "ABC")
            return (f"<div style='font-weight:600;font-size:13px;margin-top:6px'>{ad} · {ui.sayi(n)}</div><div style='display:flex;border-radius:6px;overflow:hidden;margin:4px 0'>{bar}</div>"
                    f"<div style='font-size:11px;color:#5b6475'>A {ui.sayi(c['A'])} · B {ui.sayi(c['B'])} · C {ui.sayi(c['C'])} {birim}</div>"), c, n
        t1, c1, n1 = seritler(pm, "Müşteriler", "müşteri")
        t2, c2, n2 = seritler(pu, "Ürünler", "ürün")
        st.markdown(t1 + t2, unsafe_allow_html=True)
        st.markdown(f"<div style='background:#faf7f2;border-radius:10px;padding:8px 12px;font-size:12px;margin-top:10px'>Müşterilerin %{ui.sayi(c1['A'] / n1 * 100, 1)}’i, ürünlerin %{ui.sayi(c2['A'] / n2 * 100, 0)}’i cironun %80’ini getiriyor.</div>", unsafe_allow_html=True)

    # ---------- 3. Departmanlar arası çıkarımlar (seçili tarihe göre) ----------
    churn, auc = w.churn_c(tum, vid)
    tb = pd.Timestamp(b)
    mr = an.araliga_kes(m, a, b)
    gecmis = m[m["gun"] <= tb]
    pc = an.pareto(gecmis, "CustomerID")
    c_idx = pc.index[pc["seg"] == "C"]
    son = gecmis.groupby("CustomerID")["gun"].max()
    cpas = ((tb - son.reindex(c_idx)).dt.days > 90).mean() * 100 if len(c_idx) else 0
    dow = mr.groupby("dow")["net"].sum()
    pzt = dow.get(0, 0) / dow.sum() * 100 if dow.sum() else 0
    yavas3 = c_yavas(m, K, str(b))
    with st.container(border=True):
        ui.bolum(3, "Cross-functional insights · departmanlar arası çıkarımlar", f"Tek departmana bakınca görünmeyen üç şey · {et(a, b)} dönemine göre")
        kt = [("Satış", "Pazarlama", "#d6ece5", "#fbe0d3", f"%{ui.sayi(cpas, 0)} sessiz", f"{et(b, b)} itibarıyla küçük müşterilerin (C) %{ui.sayi(cpas, 0)}’i 3 aydır sipariş vermemiş. Ciro düşüşünün sebebi fiyat değilse sipariş sayısına bak."),
              ("Satış", "Tedarik", "#d6ece5", "#dbe5f8", f"%{ui.sayi(pzt, 0)} pazartesi", "Seçili dönemde satışın bu kadarı pazartesi geliyor. Cuma gününe küçük bir indirim koyarsak depo yükü haftaya yayılır."),
              ("Tedarik", "Satış", "#dbe5f8", "#d6ece5", f"{ui.sayi(len(yavas3))} yavaş ürün", f"{et(b, b)} öncesindeki 90 günde satışı bir önceki 90 güne göre %70 ve üzeri düşen {ui.sayi(len(yavas3))} ürün var. Yanında sık alınan bir ürünle paket yapıp satabiliriz.")]
        for kol, (d1, d2, b1, b2, buyuk, txt) in zip(st.columns(3), kt):
            kol.markdown(f"<div style='background:#faf7f2;border:1px solid #e4dfd6;border-radius:12px;padding:12px 14px;height:100%'><span class='pill' style='background:{b1}'>{d1}</span>"
                         f"<span class='pill' style='background:{b2}'>{d2}</span><div style='font-family:Poppins;font-weight:700;font-size:20px;margin:6px 0'>{buyuk}</div>"
                         f"<div style='font-size:12px;line-height:1.5'>{txt}</div></div>", unsafe_allow_html=True)

    # ---------- 4. Aksiyon planı (tablo) ----------
    baz, wape = c_baz(m, K)
    yavas = c_yavas(m, K, str(bit.date()))
    cm = an.pareto(m, "CustomerID").join(churn[["p", "risk", "m30"]])
    pay = (m.groupby("CustomerID")["net"].sum() / tum.groupby("CustomerID")["net"].sum()).reindex(cm.index).fillna(0)
    cm["m30"] = cm["m30"].fillna(0) * pay
    hiAB = cm[(cm["seg"].isin(["A", "B"])) & (cm["risk"] == "Yüksek")]
    cOK = cm[(cm["seg"] == "C") & (cm["risk"].isin(["Orta", "Yüksek"]))]
    norm_top, norm_kat = c_norm(tum, m, K)
    az_say, a_say = c_abc(m, K)
    KS, KG = an2.kohort_ceyrek(m)
    gun_say = (bit - m["gun"].min()).days + 1
    tavan30 = norm_top["tavan"] * 30 / gun_say
    fiyat = m["net"].sum() / m["Quantity"].sum()
    wk_q = m.groupby("gun")["Quantity"].sum().reindex(pd.date_range(tum["gun"].min(), bit), fill_value=0)
    haf_q = wk_q.values[: len(wk_q) // 7 * 7].reshape(-1, 7).sum(axis=1)
    base_u = float(haf_q[-8:].mean())
    kapasite = float(haf_q.max())
    bitis30 = bit + pd.Timedelta(days=30)
    ss = st.session_state
    sa, sb, sc, sd = ss.get("ak_a", 25), ss.get("ak_b", 8), ss.get("ak_c", 15), ss.get("ak_d", 50)
    onA, onB, onC, onD = (ss.get(k, True) for k in ("ak_a_on", "ak_b_on", "ak_c_on", "ak_d_on"))
    oran = (1 - yavas["d"] - sb / 100) / (1 - yavas["d"])
    qa = yavas["q90"] * (oran ** -2.2 - 1)
    gA = hiAB["m30"].sum() * sa / 100
    gB = float((qa * yavas["pr"] * oran).sum()) / 3
    uB = float(qa.sum()) / 3
    gC = cOK["m30"].sum() * sc / 100
    gD = tavan30 * sd / 100
    en_buyuk = hiAB.sort_values("m30", ascending=False).head(2)
    musteri_txt = " ve ".join(f"<b>{i}</b> ({ui.tl(r['m30'])})" for i, r in en_buyuk.iterrows())
    kat2 = " ve ".join(norm_kat.index[:2].tolist()) or "ilgili kategoriler"
    PLAN = [
        ("Pazarlama", f"Bırakma riski yüksek {ui.sayi(len(hiAB))} büyük (A/B) müşteri var" + (f"; en büyükleri {musteri_txt}" if musteri_txt else "") + f". Bırakırlarsa gelecek ay ≈ {ui.tl(hiAB['m30'].sum())} kaybederiz.",
         "Müşteri temsilcisi ile doğrudan telefon/ziyaret ve özel geri kazanım sepeti.", gA, onA, "Ciro koruma", ("KRİTİK (Bu hafta)", "r")),
        ("Satış & Fiyat", f"<b>{ui.sayi(norm_top['asan_siparis'])} siparişte</b> kategori medyanından 10+ puan fazla indirim ({kat2} başta). Fazlası ≈ {ui.tl(norm_top['fazla'])}.",
         "ERP'de kategori bazlı otomatik indirim tavanı (+5 puan üstü yönetici onayı).", gD, onD, "Doğrudan brüt kâr", ("KRİTİK (Bu hafta)", "r")),
        ("Tedarik", f"{ui.sayi(len(yavas))} ürünün satışı son 90 günde %70+ düştü; <b>{ui.sayi(a_say)} A sınıfı üründen {ui.sayi(az_say)} tanesi</b> düzensiz talepli (stok riski).",
         "%95 hizmet seviyesiyle otomatik sipariş noktası; yavaş ürünleri hızlı ürünlerle ikili paket yap.", gB, onB, "Ek ciro", ("YÜKSEK (15 gün)", "y")),
        ("CRM / Satış", f"Yeni müşterilerin <b>%{ui.sayi(100 - (KG[1] or 0), 0)}’i</b> ilk siparişten sonraki çeyrekte tekrar sipariş vermedi; {ui.sayi(len(cOK))} küçük müşteride bırakma riski var.",
         "İlk faturadan 30–45 gün sonra otomatik “Birlikte alınanlar” çapraz kategori teklifi ve kupon.", gC, onC, "Ek ciro", ("ORTA (30 gün)", "g")),
    ]
    with st.container(border=True):
        ui.bolum(4, "Önümüzdeki 30 gün için aksiyon planı", f"Tahmin {et(bit + pd.Timedelta(days=1), bitis30)} için (veri {et(bit, bit)} tarihinde bitiyor). Finansal etki, aşağıdaki simülasyondaki ayarlara göre güncellenir")
        satir = []
        for birim, sorun, aksiyon_txt, g, on, alt, (oc_, oc_k) in PLAN:
            satir.append([f"<span style='display:inline-block;background:#1f2a44;color:#fff;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;white-space:nowrap'>{birim}</span>",
                          sorun, aksiyon_txt,
                          f"<b style='color:{'#14513f' if on else '#9aa3b2'}'>{'+' + ui.tl(g) if on else 'Planda değil'}</b><div style='font-size:11px;color:#5b6475'>{alt}</div>",
                          f"<span class='pill {oc_k}'>{oc_}</span>"])
        st.markdown(w.tablo(["BİRİM", "TESPİT EDİLEN KÖK SORUN", "ÖNERİLEN SOMUT AKSİYON", "HEDEF / FİNANSAL ETKİ", "ÖNCELİK"], satir), unsafe_allow_html=True)

    # ---------- 5. Etki simülasyonu (yalnız girdiler ve sonuç) ----------
    with st.container(border=True):
        ui.bolum(5, "Etki simülasyonu", "Ayarları değiştir, ek getiriyi gör")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            sa = st.slider("Geri dönen müşteri oranı (%)", 0, 60, 25, key="ak_a")
            onA = st.toggle("Planda", value=True, key="ak_a_on")
        with k2:
            sd = st.slider("Fazla indirimin kesilme oranı (%)", 0, 100, 50, key="ak_d")
            onD = st.toggle("Planda", value=True, key="ak_d_on")
        with k3:
            sb = st.slider("Pakete ek indirim (puan)", 0, 20, 8, key="ak_b")
            onB = st.toggle("Planda", value=True, key="ak_b_on")
        with k4:
            sc = st.slider("Kuponu kullanan oranı (%)", 0, 40, 15, key="ak_c")
            onC = st.toggle("Planda", value=True, key="ak_c_on")
        gA = hiAB["m30"].sum() * sa / 100
        gD = tavan30 * sd / 100
        oran = (1 - yavas["d"] - sb / 100) / (1 - yavas["d"])
        qa = yavas["q90"] * (oran ** -2.2 - 1)
        gB = float((qa * yavas["pr"] * oran).sum()) / 3
        uB = float(qa.sum()) / 3
        gC = cOK["m30"].sum() * sc / 100
        tot = (gA if onA else 0) + (gB if onB else 0) + (gC if onC else 0) + (gD if onD else 0)
        WK = 30 / 7
        ek_u = ((gA / fiyat if onA else 0) + (uB if onB else 0) + (gC / fiyat if onC else 0)) / WK
        util = (base_u + ek_u) / kapasite * 100
        fig = go.Figure()
        fig.add_bar(y=["Bir şey yapmazsak"], x=[baz / 1e6], orientation="h", marker_color="#d4cdbf", name="Tahmin", showlegend=False,
                    hovertemplate=f"<b>Bir şey yapmazsak</b><br>{ui.tl(baz)}<br>Tahmin aralığı: {ui.tl(baz * (1 - wape))} – {ui.tl(baz * (1 + wape))}<extra></extra>")
        fig.add_bar(y=["Aksiyonlarla"], x=[baz / 1e6], orientation="h", marker_color="#d4cdbf", name="Tahmin", showlegend=False, hoverinfo="skip")
        for ad, g, on, rn in (("Büyük müşterileri ara", gA, onA, "#d9622b"), ("İndirim tavanı", gD, onD, "#1f2a44"), ("Yavaş ürünleri paketle", gB, onB, "#3d9a84"), ("Kupon gönder", gC, onC, "#e0a526")):
            if on:
                fig.add_bar(y=["Aksiyonlarla"], x=[g / 1e6], orientation="h", marker_color=rn, name=ad,
                            hovertemplate=f"<b>{ad}</b><br>Ek getiri: +{ui.tl(g)}<br>Tahmine göre: +%{ui.sayi(g / baz * 100, 1)}<extra></extra>")
        fig.add_annotation(x=baz / 1e6, y="Bir şey yapmazsak", text=f"{ui.sayi(baz / 1e6, 2)} M ₺", showarrow=False, xanchor="left", xshift=6, font=dict(size=12))
        fig.add_annotation(x=(baz + tot) / 1e6, y="Aksiyonlarla", text=f"{ui.sayi((baz + tot) / 1e6, 2)} M ₺ (+%{ui.sayi(tot / baz * 100, 1)})", showarrow=False, xanchor="left", xshift=6, font=dict(size=12))
        fig.add_vrect(x0=baz * (1 - wape) / 1e6, x1=baz * (1 + wape) / 1e6, fillcolor="rgba(31,42,68,.08)", line_width=0, layer="below")
        fig.update_layout(barmode="stack", height=190, margin=dict(l=0, r=0, t=26, b=24), showlegend=True, plot_bgcolor="white",
                          legend=dict(orientation="h", y=1.22, x=0, font=dict(size=11)),
                          xaxis=dict(range=[0, baz * 1.45 / 1e6], ticksuffix=" M"), yaxis=dict(autorange="reversed"))
        r1, r2 = st.columns([2, 1])
        with r1:
            ui.plot(fig)
            st.caption(f"Gölge alan: tahmin ±%{ui.sayi(wape * 100, 0)} oynayabilir. Çubuğun üzerine gelince değerleri görürsün.")
        with r2:
            st.markdown(f"<div class='out'><div class='t'>Aksiyonlardan beklenen ek getiri</div><div class='v'>+{ui.tl(tot)} "
                        f"<span class='chg {'pos' if tot > 0 else 'mute'}' style='font-size:11px'>+%{ui.sayi(tot / baz * 100, 1)}</span></div></div>"
                        f"<div style='margin-top:6px'><span class='pill' style='background:{'#fbe3e0' if util > 100 else ('#fdf0cf' if util > 90 else '#d6ece5')}'>Stok kapasitesi %{ui.sayi(util, 0)} dolu</span></div>"
                        f"<div style='font-size:12px;margin-top:6px'>Hiçbir şey yapmazsan: <b>{ui.tl(baz)}</b><br>Aksiyonlarla: <b>{ui.tl(baz + tot)}</b></div>", unsafe_allow_html=True)

    # ---------- RafBot ----------
    aksiyon = sorted([("Büyük müşterileri ara", gA if onA else 0), ("İndirim tavanı", gD if onD else 0), ("Yavaş ürünleri paketle", gB if onB else 0), ("Küçük müşterilere kupon gönder", gC if onC else 0)], key=lambda x: -x[1])[0]
    bilgi = {
        "ciro": f"Önümüzdeki 30 gün için baz tahmin ≈ {ui.tl(baz)}. Geriye dönük testte ortalama hata ≈ %{ui.sayi(wape * 100, 0)}, yani tahmin ±{ui.tl(baz * wape)} aralığında okunmalı.",
        "kopru": f"Net ciro {et(a, b)} döneminde {ui.tl(S['net'])}; karşılaştırma döneminden %{ui.sayi(abs(w.rel(S['net'], C['net']) * 100), 1)} {'fazla' if S['net'] >= C['net'] else 'az'}.",
        "churn": f"{ui.sayi(len(hiAB))} büyük müşteri yüksek riskli; bırakırlarsa gelecek ay ≈ {ui.tl(hiAB['m30'].sum())} kaybederiz. Önce bunları ara.",
        "segment": f"Müşterilerin %{ui.sayi(c1['A'] / n1 * 100, 1)}’i, ürünlerin %{ui.sayi(c2['A'] / n2 * 100, 0)}’i cironun %80’ini getiriyor.",
        "indirim": f"Şu anki ayarlarla en çok getiren aksiyon “{aksiyon[0]}” (+{ui.tl(aksiyon[1])}). Dördünün toplamı +{ui.tl(tot)}, yani tahminin %{ui.sayi(tot / baz * 100, 1)}’i.",
        "gun": f"Cironun %{ui.sayi(pzt, 0)}’i pazartesi geliyor.", "birlikte": None,
    }
    chat.rafbot(bilgi)


# ======================================================================
# SAYFA: Satış & Fiyat
# ======================================================================
def sayfa_satis():
    GUNLER = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    KAMPANYA = {
        "musteri": {
            "A": ("Sadık hesap:", "hesap yöneticisi + hacim kademeli indirim. Haftanın en yoğun gününü azaltmak için ek indirim denenebilir."),
            "B": ("A'ya yükselt:", "sepet tutarı hedefi koyup eşiği geçen siparişe sonraki siparişte indirim ver."),
            "C": ("Aktifleştir, ucuza:", "\"birlikte al\" önerisi + tekrar sipariş kuponu; riskli olanlara otomatik e-posta/SMS. Yüksek maliyetli aksiyon değmez."),
        },
        "urun": {
            "A": ("Yıldız ürün:", "stoktan eksik etme, indirimi artırma. \"Birlikte al\" önerisinde ana ürün olarak kullan."),
            "B": ("A'ya taşı:", "yıldız ürünlerle paket yap, kategori kampanyalarında öne çıkar."),
            "C": ("Uzun kuyruk:", "satılmayanları hızlı bir ürünle paketleyip erit; bu ürünlerde yeni stok alma."),
        },
    }
    SEG_AD = {"A": "A · cironun %80'i", "B": "B · sonraki %15", "C": "C · son %5"}
    SEG_RENK = {"A": ("#d6ece5", "#14513f"), "B": ("#fdf0cf", "#7a5600"), "C": ("#fbe0d3", "#7d2f0e")}
    HEAT = {("A", "Düşük"): ("#2f7f6b", "#fff"), ("A", "Orta"): ("#fdf0cf", "#1f2a44"), ("A", "Yüksek"): ("#fbe3e0", "#1f2a44"),
            ("B", "Düşük"): ("#2f7f6b", "#fff"), ("B", "Orta"): ("#f6d98a", "#1f2a44"), ("B", "Yüksek"): ("#f2b8b2", "#1f2a44"),
            ("C", "Düşük"): ("#8cc5b5", "#1f2a44"), ("C", "Orta"): ("#f0c250", "#1f2a44"), ("C", "Yüksek"): ("#a8322b", "#fff")}
    RISK_PILL = {"Yüksek": "r", "Orta": "y", "Düşük": "g"}


    @st.cache_data(show_spinner=False)
    def c_ozet(_m, anahtar, a, b):
        return an.ozet(_m, a, b)


    @st.cache_data(show_spinner="Birlikte alınanlar hesaplanıyor…")
    def c_birlikte(_m, anahtar, a, b):
        x = an.araliga_kes(_m, a, b)
        return an.birlikte(x), an.en_cok_satanlar(x)


    @st.cache_data(show_spinner="Birlikte alınanlar (tüm yıl) hesaplanıyor…")
    def c_birlikte_yil(_m, anahtar, yil):
        return an.birlikte(_m[_m["gun"].dt.year == yil])


    @st.cache_data(show_spinner="Churn modeli eğitiliyor…")
    def c_churn(_df, vid):
        return an.churn_modeli(_df)


    @st.cache_data(show_spinner=False)
    def c_supheli(_tum, _m, anahtar, a, b, tur):
        x, oc = an.supheli(_tum, an.araliga_kes(_m, a, b), tur)
        return x, int((oc["sapma"] > 0.10).sum()), float(oc.loc[oc["sapma"] > 0.10, "fazla"].sum())


    def pct(x):
        return ("▲ %" if x >= 0 else "▼ %") + ui.sayi(abs(x * 100), 1)


    def chg(x, vurgu):
        sinif = "mute" if not vurgu else ("neg" if x < 0 else "pos")
        return f"<span class='chg {sinif}'>{pct(x)}</span>"


    def chg_p(d, vurgu):
        return f"<span class='chg {'neu' if vurgu else 'mute'}'>{'▲ ' if d >= 0 else '▼ '}{ui.sayi(abs(d * 100), 1)} puan</span>"


    def rel(a, b):
        return a / b - 1 if b else 0


    def kart(baslik, nom, nom_c, gun_et, gun_v, gun_c, sari=False):
        return (f"<div class='kpi{' y' if sari else ''}'><div class='kl'>{baslik}</div><div class='kv'>{nom}</div>{nom_c}"
                f"<div class='rule'></div><div class='kl'>{gun_et}</div><div class='kv2'>{gun_v}</div>{gun_c}</div>")


    def tablo(basliklar, satirlar, sag=()):
        h = "".join(f"<th class='{'r' if i in sag else ''}'>{b}</th>" for i, b in enumerate(basliklar))
        r = "".join("<tr>" + "".join(f"<td class='{'r' if i in sag else ''}'>{c}</td>" for i, c in enumerate(s)) + "</tr>"
                    for s in satirlar)
        return f"<table class='t'><thead><tr>{h}</tr></thead><tbody>{r}</tbody></table>"


    def isi_rengi(v, mx):
        t = 0 if mx <= 0 else min(1, v / mx)
        a, b = (238, 233, 223), (27, 94, 75)
        c = tuple(int(a[i] + (b[i] - a[i]) * t ** 0.8) for i in range(3))
        yazi = "#fff" if t > 0.55 else "#1f2a44"
        return f"background:rgb{c};color:{yazi}"


    tum = data.veri()
    vid = (st.session_state.get("dosya_adi", ""), len(tum))
    ui.baslik("Satış & Fiyat", "Ne zaman, kime, hangi ürünü, hangi indirimle satmalıyız?")
    m, a, b, ca, cb = ui.filtreler(tum)
    fk = (vid, st.session_state.get("f_kat"), st.session_state.get("f_urun"), st.session_state.get("f_sehir"))

    if m.empty:
        st.warning("Bu filtrelerle veri yok. Filtreleri gevşetip tekrar dene.")
        st.stop()

    S = c_ozet(m, fk, a, b)
    C = c_ozet(m, fk, ca, cb)
    ayni = S["gun"] == C["gun"]
    hN, hD = ayni, not ayni
    et = ui.tarih_etiketi

    st.markdown(
        f"<span class='pill g'>Seçili: {et(a, b)} · {S['gun']} gün</span><span style='font-size:12px;color:#5b6475'>karşılaştırma</span> "
        f"<span class='pill'>{et(ca, cb)} · {C['gun']} gün</span>"
        f"<span class='pill y'>{'Aralıklar eşit uzunlukta: vurgulu fark nominal değerlere göre' if ayni else 'Aralık uzunlukları farklı: vurgulu fark günlük ortalamaya göre'}</span>"
        "<div class='kgrid'>" +
        kart("Net ciro", ui.sayi(S["net"] / 1e6, 2) + " M ₺", chg(rel(S["net"], C["net"]), hN), "Günlük ort.",
             ui.sayi(S["dnet"] / 1000, 1) + " bin ₺", chg(rel(S["dnet"], C["dnet"]), hD)) +
        kart("Sipariş adedi", ui.sayi(S["ord"]), chg(rel(S["ord"], C["ord"]), hN), "Günlük ort.",
             ui.sayi(S["dord"], 1), chg(rel(S["dord"], C["dord"]), hD)) +
        kart("Ort. sepet adedi", ui.sayi(S["b"]) + " adet", chg(rel(S["b"], C["b"]), hN), "Günlük ort.",
             ui.sayi(S["dbq"]) + " adet", chg(rel(S["dbq"], C["dbq"]), hD)) +
        kart("Ort. sepet tutarı", ui.sayi(S["basket"], 1) + " ₺", chg(rel(S["basket"], C["basket"]), hN), "Günlük ort.",
             ui.sayi(S["dbt"], 1) + " ₺", chg(rel(S["dbt"], C["dbt"]), hD)) +
        kart("İndirim oranı", "%" + ui.sayi(S["disc"] * 100, 1), chg_p(S["disc"] - C["disc"], hN), "Günlük ort.",
             "%" + ui.sayi(S["dd"] * 100, 1), chg_p(S["dd"] - C["dd"], hD), sari=True) +
        "</div><p class='sub'>Aralıklar eşit uzunlukta olursa (ör. 10 gün – 10 gün) vurgu nominal değerlere geçer. "
        "Aralıklar farklıysa (ör. 10 gün – 1 ay) günlük ortalamalar kıyaslanır.</p>", unsafe_allow_html=True)

    # ---------- 1. Günlük ciro neden değişti? ----------
    ef = [v / 1000 for v in an.kopru(S, C)]
    adlar = ["Sipariş sayısı", "Sepet adedi", "Ürün karması", "İndirim"]
    altlar = [f"günlük sipariş {ui.sayi(C['dord'], 1)} → {ui.sayi(S['dord'], 1)}",
              f"sepette {ui.sayi(C['b'], 1)} → {ui.sayi(S['b'], 1)} adet",
              f"ort. liste fiyat {ui.sayi(C['L'], 2)} → {ui.sayi(S['L'], 2)} ₺",
              f"indirim %{ui.sayi(C['disc'] * 100, 1)} → %{ui.sayi(S['disc'] * 100, 1)}"]
    dch = rel(S["dnet"], C["dnet"])
    dom = int(np.argmax([abs(v) for v in ef]))
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        with h1:
            ui.bolum(1, "Günlük ciro neden değişti?", "Seçili dönem ile karşılaştırma dönemi arasındaki günlük ciro farkı, dört nedene bölündü")
        with h2:
            st.markdown(f"<div style='text-align:right;padding-top:6px'><span class='pill'>{ui.sayi(C['dnet'] / 1000, 1)} bin ₺</span> → "
                        f"<span class='pill g'>{ui.sayi(S['dnet'] / 1000, 1)} bin ₺</span> "
                        f"<span class='chg {'neg' if dch < 0 else 'pos'}'>{pct(dch)}</span></div>", unsafe_allow_html=True)
        s1, s2 = st.columns([3, 2])
        with s1:
            fig = go.Figure(go.Bar(
                y=[f"{n}<br><span style='font-size:10px;color:#5b6475'>{s}</span>" for n, s in zip(adlar, altlar)],
                x=ef, orientation="h", marker_color=[ui.RENK["kirmizi"] if v < 0 else ui.RENK["yesil"] for v in ef],
                text=[f"{ui.isaret(v, 1)} bin ₺" for v in ef], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{text} günlük ciro etkisi<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            mxa = max([abs(v) for v in ef] + [1])
            fig.update_xaxes(zeroline=True, zerolinecolor="#9aa3b2", showgrid=False, showticklabels=False,
                             range=[-mxa * 1.45, mxa * 0.45 if min(ef) < 0 and max(ef) <= 0 else mxa * 1.45])
            fig.update_yaxes(automargin=True)
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=6, b=0), plot_bgcolor="white")
            ui.plot(fig)
        with s2:
            yil_hizi = tum["net"].sum() / ((tum["gun"].max() - tum["gun"].min()).days + 1)
            st.markdown(f"<div class='out'><div class='t'>Bu hızla sürerse (30 gün)</div><div class='v'>{ui.sayi(S['dnet'] * 30 / 1e6, 2)} M ₺</div>"
                        f"<div class='t'>veri ortalaması hızı: {ui.sayi(yil_hizi * 30 / 1e6, 2)} M ₺ · tahmin değil, hız çarpımı</div></div>",
                        unsafe_allow_html=True)
            if dom == 3:
                b1 = f"**İndirim en büyük etki.** Günlük ciroda {ui.isaret(ef[3], 1)} bin ₺ fark yarattı; indirimi hedefli kullan (bkz. 2)."
            elif dom == 2:
                b1 = f"**Ürün karmasına bak.** Satılan ürünlerin liste fiyatı değişti ({ui.isaret(ef[2], 1)} bin ₺)."
            else:
                b1 = (f"**{'İndirimi artırma.' if dch < 0 else 'Fiyat disiplinini koru.'}** Değişim fiyattan değil hacimden: "
                      f"indirim etkisi {ui.isaret(ef[3], 1)} bin ₺.")
            st.markdown("**Ne yapmalı?**")
            st.markdown(f"1. {b1}\n2. **Sipariş sıklığını artır.** Pasif C müşterilerini aktifleştir (bkz. 6 ve 7).\n"
                        "3. **Sepeti büyüt.** Birlikte al önerisini sipariş ekranına koy (bkz. 4).")

    # ---------- 2. İndirim planı ----------
    urunler = an.urun_listesi(tum, 50)
    with st.container(border=True):
        ui.bolum(2, "İndirim planı", f"Hedefler seçili dönem için ({et(a, b)}, {S['gun']} gün). Karşılaştırma dönemi: {et(ca, cb)}")
        pl, pr = st.columns([3, 2])
        with pl:
            urun = st.selectbox("Ürün", urunler, key="plan_urun")
            P = an.urun_referans(tum, urun, S["gun"], ca, cb)
            Qr, Nr, Gr = P["q"], P["n"], P["g"]
            L = Gr / Qr if Qr else 0
            d0 = 1 - Nr / Gr if Gr else 0
            dq = st.slider(f"Hedef satış (adet) · karşılaştırma: {ui.sayi(Qr)}", -50, 100, 10, format="%d%%", key="plan_q")
            dc = st.slider(f"Hedef ciro (₺) · karşılaştırma: {ui.sayi(Nr)}", -50, 100, 5, format="%d%%", key="plan_c")
            cp = st.slider("Alış maliyeti (liste fiyatın yüzdesi)", 0, 100, 44, format="%d%%", key="plan_m",
                           help="Veride maliyet yok; bu ürünün maliyetini biliyorsan değiştir.")
            Cc = cp / 100
            Kr = Nr - Cc * Gr
            dk = st.slider(f"Hedef kâr (₺) · karşılaştırma: {ui.sayi(Kr)}", -50, 100, 0, format="%d%%", key="plan_k")
            st.caption(f"Liste fiyatı {ui.sayi(L, 2)} ₺ · hedef satış {ui.sayi(Qr * (1 + dq / 100))} adet · "
                       f"hedef ciro {ui.sayi(Nr * (1 + dc / 100))} ₺ · hedef kâr {ui.sayi(Kr * (1 + dk / 100))} ₺")
        tq, tc, tk = Qr * (1 + dq / 100), Nr * (1 + dc / 100), Kr * (1 + dk / 100)
        yeni = 1 - (tc / tq) / L if tq > 0 and L > 0 else 0
        kar = tc - Cc * L * tq
        nedenler = []
        if yeni > P["cap"] + 0.0005:
            nedenler.append("Gereken indirim ürünün yıl içinde yaygın üst sınırını aşıyor.")
        if kar < tk - 0.5:
            nedenler.append("Bu fiyat ve adetle kâr hedefi tutmuyor.")
        tamam = not nedenler
        with pr:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='out'><div class='t'>Gereken indirim</div><div class='v'>%{ui.sayi(yeni * 100, 1)}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='out'><div class='t'>Karşılaştırma dönemi</div><div class='v'>%{ui.sayi(d0 * 100, 1)}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='out'><div class='t'>Fark</div><div class='v'>{ui.isaret((yeni - d0) * 100, 1)} puan</div></div>", unsafe_allow_html=True)
            durum = ("<span class='pill g'><b>ULAŞILABİLİR</b></span>" if tamam else
                     "<span class='pill' style='background:#e0a526'><b>HEDEF ZOR</b></span>")
            st.markdown(f"{durum} {'Hedefler birlikte tutuyor.' if tamam else ' '.join(nedenler)}", unsafe_allow_html=True)
            st.markdown(f"Bu hedefte tahmini kâr: **{ui.sayi(kar)} ₺** (hedef: {ui.sayi(tk)} ₺). Ürün yıl içinde satışlarının "
                        f"%95'ini en fazla %{ui.sayi(P['cap'] * 100)} indirimle yaptı, en yüksek indirim %{ui.sayi(P['mx'] * 100)}.")
            if P["fb"]:
                st.caption("Karşılaştırma döneminde bu ürün satılmamış; referans olarak veri ortalaması günlük hız kullanıldı. "
                           "İndirim = 1 − (hedef ciro ÷ hedef adet) ÷ liste fiyatı.")
            else:
                st.caption(f"Karşılaştırma dönemi {C['gun']} gün, seçili dönem {S['gun']} gün: karşılaştırma değerleri günlük ortalamayla "
                           f"{S['gun']} güne çevrildi. İndirim = 1 − (hedef ciro ÷ hedef adet) ÷ liste fiyatı.")

    # ---------- 3. Haftanın günü × kategori ----------
    mr = an.araliga_kes(m, a, b)
    with st.container(border=True):
        t1, t2 = st.columns([3, 1])
        with t1:
            ui.bolum(3, "Haftanın günü × kategori", "Her kategorinin cirosunun günlere dağılımı (%) · en çok satan 6 kategori")
        with t2:
            gos = st.selectbox("Gösterim", ["Ciro payı", "Sipariş adedi"], key="hk_gos")
        hk = an.hafta_kategori(mr)[gos] if not mr.empty else pd.DataFrame()
        if hk.empty:
            st.info("Bu aralıkta satış yok.")
            gun_cum = ""
        else:
            mx = float(hk.values.max())
            satir = [[f"<td style='text-align:left'>{k}</td>"] + [
                f"<td style='text-align:center;{isi_rengi(v, mx)}'>{ui.sayi(v, 1)}</td>" for v in hk.loc[k].values] for k in hk.index]
            st.markdown("<table class='t'><thead><tr><th>Kategori</th>" + "".join(f"<th style='text-align:center'>{g}</th>" for g in GUNLER) +
                        "</tr></thead><tbody>" + "".join("<tr>" + "".join(r) + "</tr>" for r in satir) + "</tbody></table>", unsafe_allow_html=True)
            dow = mr.groupby("dow")["net"].sum().reindex(range(7), fill_value=0)
            pay = dow / dow.sum() * 100
            en = np.unravel_index(np.argmax(hk.values), hk.shape)
            en_gun = ["pazartesiye", "salıya", "çarşambaya", "perşembeye", "cumaya", "cumartesiye", "pazara"][en[1]]
            gun_cum = (f"Cironun %{ui.sayi(pay.max(), 0)}'i {['pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar'][int(pay.idxmax())]} günü, "
                       f"%{ui.sayi(pay[5] + pay[6], 1)}'si hafta sonu. {hk.index[en[0]]} {en_gun} yığılmış (%{ui.sayi(hk.values[en], 1)}). "
                       "Veride saat bilgisi yok, yalnız gün var.")
            st.caption(gun_cum)

    # ---------- 4. Birlikte alınanlar ----------
    cift, cok = c_birlikte(m, fk, a, b)
    yil = pd.Timestamp(b).year
    cift_yil = c_birlikte_yil(m, fk, yil)
    with st.container(border=True):
        ui.bolum(4, "Birlikte alınanlar ve en çok satanlar", "Aynı siparişte en sık birlikte geçen ürün çiftleri · seçilen tarih aralığı ve yıl")
        k1, k2 = st.columns(2)
        for kol, bas, alt, liste in ((k1, "Seçilen tarih", et(a, b), cift), (k2, "Tüm yıl", f"Seçilen tarihin yılı: {yil}", cift_yil)):
            with kol, st.container(border=True):
                st.markdown(f"**{bas}**")
                st.caption(alt)
                if not liste:
                    st.write("Çift bulunamadı.")
                if liste:
                    st.markdown("<ol class='lst'>" + "".join(
                        f"<li><b>{c['a']} + {c['b']}</b><br><small>{c['n']} sipariş · %{ui.sayi(c['oran'] * 100, 0)}</small></li>"
                        for c in liste) + "</ol>", unsafe_allow_html=True)
        st.markdown(f"**Seçilen tarihte best seller'lar ve en çok birlikte alındıkları ürün · {et(a, b)}**")
        if cok.empty:
            st.info("Bu aralıkta satış yok.")
        else:
            g = cok.copy()
            g["Birlikte alma oranı"] = g["Birlikte alma oranı"].map(lambda v: f"%{ui.sayi(v * 100, 0)}")
            g["#"] = g["#"].astype(str)
            g["Adet"] = g["Adet"].map(ui.sayi)
            st.dataframe(g, hide_index=True, use_container_width=True)
        st.caption("Birlikte alma oranı: best seller'ı alan siparişlerin yüzde kaçında bu ürün de var. "
                   "Sepette best seller varsa ikinci ürün öneri olarak gösterilir.")

    # ---------- 5-6. Pareto ve kampanya ----------
    _sk, _ = st.columns([1, 2])
    tur = _sk.selectbox("Segmentasyon türü", ["Müşteri segmentasyonu", "Ürün segmentasyonu"], key="seg_tur")
    musteri_mi = tur.startswith("Müşteri")
    anahtar = "CustomerID" if musteri_mi else "Description"
    birim = "Müşteri" if musteri_mi else "Ürün"
    tip = "musteri" if musteri_mi else "urun"
    pc = an.pareto(mr, anahtar)
    churn, auc = c_churn(tum, vid)
    if pc.empty:
        st.info("Bu aralıkta satış yok.")
        st.stop()
    n = len(pc)
    sayilar = {s: int((pc["seg"] == s).sum()) for s in "ABC"}
    paylar = {s: pc.loc[pc["seg"] == s, "net"].sum() / pc["net"].sum() for s in "ABC"}

    if musteri_mi:
        ozet_m = an.musteri_ozeti(tum, mr, pc, tum["gun"].max())
        ozet_m = ozet_m.join(churn[["p", "risk"]])
    else:
        son_satis = tum.groupby("Description")["gun"].max()
        ozet_m = pc.copy()
        g = mr.groupby("Description").agg(adet=("Quantity", "sum"), brut=("gross", "sum"))
        ozet_m["adet"] = g["adet"]
        ozet_m["indirim"] = 1 - ozet_m["net"] / g["brut"]
        ozet_m["son"] = son_satis.reindex(ozet_m.index)
        ozet_m["pasif"] = (tum["gun"].max() - ozet_m["son"]).dt.days

    p5, p6 = st.columns([2, 3])
    with p5, st.container(border=True):
        ui.bolum(5, f"{birim} Pareto segmentleri", f"{birim}ler ciro payına göre sıralanıp 3 gruba bölündü: A %80, B %15, C %5")
        x = np.arange(1, n + 1) / n * 100
        fig = go.Figure()
        sinir = [0] + [sayilar["A"] / n * 100, (sayilar["A"] + sayilar["B"]) / n * 100, 100]
        for (x0, x1, renk, ad) in ((sinir[0], sinir[1], "#d6ece5", "A"), (sinir[1], sinir[2], "#fdf0cf", "B"), (sinir[2], sinir[3], "#fbe0d3", "C")):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=renk, line_width=0, layer="below", annotation_text=ad, annotation_position="bottom")
        fig.add_scatter(x=np.r_[0, x], y=np.r_[0, pc["kum"].values * 100], mode="lines", line=dict(color="#1f2a44", width=2),
                        hovertemplate="%{x:.1f}% → %{y:.1f}%<extra></extra>")
        fig.add_scatter(x=[sinir[1]], y=[80], mode="markers+text", marker=dict(color="#c8503f", size=9),
                        text=[f"%{ui.sayi(sinir[1], 1)} → %80"], textposition="middle right")
        fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title=f"{birim.lower()} yüzdesi (ciroya göre sıralı)", yaxis_title="kümülatif ciro %",
                          xaxis_range=[0, 100], yaxis_range=[0, 105], plot_bgcolor="white")
        ui.plot(fig)

    with p6, st.container(border=True):
        ui.bolum(6, "Segmente özel kampanya önerisi" if musteri_mi else "Ürüne özel kampanya önerisi",
                 "Her segment için ayrı hedef ve aksiyon · kampanya fikirleri A/B testle doğrulanmalı")
        satirlar = []
        for s in "ABC":
            z = ozet_m[ozet_m["seg"] == s]
            if musteri_mi:
                beh = f"Medyan {ui.sayi(z['siparis'].median(), 0)} sipariş, %{ui.sayi((z['pasif'] > 90).mean() * 100, 0)}'si 90+ gün pasif"
                if s == "C":
                    beh += f" · müşteri başı ≈ {ui.sayi(z['net'].mean(), 0)} ₺"
            else:
                beh = (f"Ürün başı ≈ {ui.sayi(z['net'].mean(), 0)} ₺, ort. indirim %{ui.sayi(z['indirim'].mean() * 100, 1)}, "
                       f"%{ui.sayi((z['pasif'] > 120).mean() * 100, 0)}'i 120+ gündür satılmıyor")
            bg, fg = SEG_RENK[s]
            lead, rest = KAMPANYA[tip][s]
            satirlar.append([f"<span class='pill' style='background:{bg};color:{fg}'>{SEG_AD[s]}</span>", ui.sayi(sayilar[s]),
                             f"%{ui.sayi(paylar[s] * 100, 1)}", beh, f"<b>{lead}</b> {rest}"])
        st.markdown(tablo(["Segment", birim, "Ciro payı", "Davranış", "Önerilen kampanya"], satirlar, sag=(1, 2)), unsafe_allow_html=True)
        if musteri_mi:
            ex = pd.DataFrame({
                "Müşteri no": ozet_m.index, "Segment": ozet_m["seg"].map(SEG_AD).values, "İl": ozet_m["il"].values,
                "Sipariş sayısı": ozet_m["siparis"].values, "Son sipariş": ozet_m["son"].dt.strftime("%d.%m.%Y").values,
                "Kaç gündür sipariş yok": ozet_m["pasif"].values, "Churn riski": ozet_m["risk"].astype(str).values,
                "Net ciro (₺)": ozet_m["net"].round(0).values,
                "Önerilen kampanya": [" ".join(KAMPANYA["musteri"][s]) for s in ozet_m["seg"]]})
        else:
            ex = pd.DataFrame({
                "Ürün adı": ozet_m.index, "Segment": ozet_m["seg"].map(SEG_AD).values, "Satılan adet": ozet_m["adet"].values,
                "İndirim oranı": ozet_m["indirim"].round(3).values, "Son satış günü": ozet_m["son"].dt.strftime("%d.%m.%Y").values,
                "Net ciro (₺)": ozet_m["net"].round(0).values,
                "Önerilen kampanya": [" ".join(KAMPANYA["urun"][s]) for s in ozet_m["seg"]]})
        st.download_button(f"Excel'e aktar: tüm {birim.lower()}ler", ui.xlsx(ex, birim), file_name=f"{tip}_segmentleri.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.caption("Listede her " + ("müşteri için: müşteri no · segment · il · sipariş sayısı · son sipariş · churn riski · önerilen kampanya."
                                     if musteri_mi else "ürün için: ürün adı · segment · satılan adet · indirim oranı · son satış günü · önerilen kampanya."))

    # ---------- 7-8. Churn ısı haritası ----------
    cm = an.pareto(mr, "CustomerID").join(churn[["p", "risk", "m30"]]).join(
        an.musteri_ozeti(tum, mr, an.pareto(mr, "CustomerID"), tum["gun"].max())[["il", "siparis", "son", "pasif"]])
    cm["risk"] = cm["risk"].astype(str)
    RISKLER = ["Düşük", "Orta", "Yüksek"]
    RK = {"Düşük": "dusuk", "Orta": "orta", "Yüksek": "yuksek"}
    ss = st.session_state
    ss.setdefault("hm", ("A", "Yüksek"))
    css = ""
    for (sg, rk), (bg, fg) in HEAT.items():
        sec = ss["hm"] == (sg, rk)
        css += (f".st-key-hm_{sg}_{RK[rk]} button{{background:{bg};color:{fg};border:none;border-radius:6px;min-height:48px;font-weight:700;"
                f"{'outline:3px solid #1f2a44;outline-offset:-3px;' if sec else ''}}}")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    h7, h8 = st.columns(2)
    with h7, st.container(border=True):
        ui.bolum(7, "Churn risk ısı haritası", "Önümüzdeki 30 günde sipariş vermeme olasılığı · bir kutuya tıkla, sağda o kutudaki müşteriler görünsün")
        bas = st.columns([1.4, 1, 1, 1])
        bas[0].markdown("<span class='sub'>Segment</span>", unsafe_allow_html=True)
        for i, rk in enumerate(RISKLER):
            bas[i + 1].markdown(f"<span class='sub'>{rk} risk</span>", unsafe_allow_html=True)
        for sg in "ABC":
            sat = st.columns([1.4, 1, 1, 1])
            sat[0].markdown(f"<div style='padding-top:10px;font-size:13px'>{SEG_AD[sg]}</div>", unsafe_allow_html=True)
            for i, rk in enumerate(RISKLER):
                adet = int(((cm["seg"] == sg) & (cm["risk"] == rk)).sum())
                if sat[i + 1].button(str(adet), key=f"hm_{sg}_{RK[rk]}", use_container_width=True):
                    ss["hm"] = (sg, rk)
                    st.rerun()
        toplam_net = cm["net"].sum()
        pills = ""
        for rk in RISKLER:
            z = cm[cm["risk"] == rk]
            pills += f"<span class='pill {RISK_PILL[rk]}'>{rk}: {ui.sayi(len(z))} müşteri · ciro %{ui.sayi(z['net'].sum() / toplam_net * 100, 1)}</span>"
        st.markdown(pills, unsafe_allow_html=True)
        st.caption(f"Gradient Boosting · zaman sıralı testte AUC {ui.sayi(auc, 2) if auc else 'hesaplanamadı'} · risk: yüksek ≥ %70, orta %40–70. "
                   "Model tüm veriyle eğitilir, risk veri bitişi itibarıyla hesaplanır.")

    sg, rk = ss["hm"]
    hucre = cm[(cm["seg"] == sg) & (cm["risk"] == rk)].sort_values("net", ascending=False)
    with h8, st.container(border=True):
        ui.bolum(8, f"{sg} · {rk} risk: en yüksek cirolu müşteriler", f"{ui.sayi(len(hucre))} müşteriden ciroya göre ilk 5 · tümü Excel'e iner")
        satirlar = [[i, ui.tl(r["net"]), f"{int(r['pasif']) if pd.notna(r['pasif']) else '—'} gün",
                     f"<span class='pill {RISK_PILL[rk]}'>%{ui.sayi(r['p'] * 100, 0)}</span>"] for i, r in hucre.head(5).iterrows()]
        st.markdown(tablo(["Müşteri", "Ciro", "Son siparişten", "Risk"], satirlar, sag=(1, 2, 3)), unsafe_allow_html=True)
        liste = pd.DataFrame({
            "Müşteri": hucre.index, "İl": hucre["il"].values, "Segment (A-B-C)": sg, "Churn riski": rk,
            "Bırakma olasılığı (%)": (hucre["p"] * 100).round(0).values, "Net ciro (₺)": hucre["net"].round(0).values,
            "Sipariş sayısı": hucre["siparis"].values, "Son sipariş": hucre["son"].dt.strftime("%d.%m.%Y").values,
            "Kaç gündür sipariş yok": hucre["pasif"].values,
            "Önümüzdeki 30 gün beklenen ciro (₺)": hucre["m30"].round(0).values})
        st.download_button(f"Excel'e indir: {sg} · {rk} risk ({ui.sayi(len(hucre))} müşteri)", ui.xlsx(liste, "Müşteriler"),
                           file_name=f"musteriler_{sg}_{RK[rk]}_risk.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.caption({"Yüksek": "Öneri: bu müşterilere kişisel arama; C segmentinde otomatik hatırlatma.",
                    "Orta": "Öneri: sipariş aralığına göre hatırlatma gönder.",
                    "Düşük": "Öneri: düzenli alıyorlar; çapraz satış teklifi uygun."}[rk] + " Ciro: seçili dönem net ciro.")

    # ---------- 9. Şüpheli sipariş radarı ----------
    with st.container(border=True):
        r1, r2 = st.columns([3, 1])
        with r1:
            ui.bolum(9, "Şüpheli sipariş radarı", "Normal dışı sipariş tespiti · neye göre sapanı seçin, liste ona göre değişir")
        with r2:
            sapma = st.selectbox("Sapma türü", ["İndirime göre sapanlar", "Hacme göre sapanlar"], key="sup_tur")
        tkey = "indirim" if sapma.startswith("İndirim") else "hacim"
        sup, n_ind, kayip = c_supheli(tum, m, fk, a, b, tkey)
        st.markdown(f"<span class='pill r'>{ui.sayi(n_ind)} sipariş-kategoride kategori medyanından 10+ puan fazla indirim</span>"
                    f"<span class='pill y'>Toplam fazla indirim ≈ {ui.tl(kayip)}</span>", unsafe_allow_html=True)
        if sup.empty:
            st.info("Bu aralıkta bu türde şüpheli sipariş yok.")
        else:
            if tkey == "indirim":
                tb = pd.DataFrame({"Sipariş": sup["InvoiceNo"], "Müşteri": sup["musteri"], "Kategori": sup["Category"],
                                   "İndirim": sup["indirim"].map(lambda v: f"%{ui.sayi(v * 100, 0)}"),
                                   "Kategori medyan indirimi": sup["md"].map(lambda v: f"%{ui.sayi(v * 100, 0)}"),
                                   "Sapma": sup["sapma"].map(lambda v: f"+{ui.sayi(v * 100, 0)} puan"),
                                   "Fazla indirim (₺)": sup["fazla"].round(0)})
            else:
                tb = pd.DataFrame({"Sipariş": sup["InvoiceNo"], "Müşteri": sup["musteri"], "Kategori": sup["Category"],
                                   "Adet": sup["q"], "Kategori medyan adedi": sup["mq"].round(0),
                                   "Katı": sup["qr"].map(lambda v: f"{ui.sayi(v, 0)}x"), "Net ciro (₺)": sup["n"].round(0)})
            st.dataframe(tb.head(10), hide_index=True, use_container_width=True)
            st.download_button("Excel'e aktar", ui.xlsx(tb, "Şüpheli siparişler"), file_name=f"supheli_siparisler_{tkey}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    # ---------- RafBot ----------
    yuksek = cm[cm["risk"] == "Yüksek"]
    a_yuk = cm[(cm["seg"] == "A") & (cm["risk"] == "Yüksek")]
    bilgi = {
        "kopru": (f"Seçili dönemde günlük ciro karşılaştırma dönemine göre {'düştü' if dch < 0 else 'arttı'} (%{ui.sayi(abs(dch) * 100, 1)}; "
                  f"{ui.sayi(C['dnet'] / 1000, 1)} → {ui.sayi(S['dnet'] / 1000, 1)} bin ₺). En büyük etki: {adlar[dom]} ({ui.isaret(ef[dom], 1)} bin ₺)."),
        "churn": (f"{ui.sayi(len(yuksek))} müşteri yüksek riskli (30 günde sipariş vermeme olasılığı ≥ %70). A segmentinde {ui.sayi(len(a_yuk))} "
                  f"yüksek riskli müşteri var; önce bunları ara" + (f" ({', '.join(map(str, a_yuk.sort_values('net', ascending=False).head(3).index))})." if len(a_yuk) else ".")),
        "segment": (f"Müşteriler ciroya göre üç gruba ayrıldı: A {ui.sayi(sayilar['A'])} (cironun %{ui.sayi(paylar['A'] * 100, 0)}'i), "
                    f"B {ui.sayi(sayilar['B'])} (%{ui.sayi(paylar['B'] * 100, 0)}), C {ui.sayi(sayilar['C'])} (%{ui.sayi(paylar['C'] * 100, 0)})."
                    if musteri_mi else f"Ürünler üç gruba ayrıldı: A {ui.sayi(sayilar['A'])}, B {ui.sayi(sayilar['B'])}, C {ui.sayi(sayilar['C'])} ürün."),
        "indirim": (f"Seçili dönemde ağırlıklı indirim oranı %{ui.sayi(S['disc'] * 100, 1)}. {ui.sayi(n_ind)} sipariş-kategoride kategori medyanından "
                    f"10+ puan fazla indirim verilmiş; toplam fazlası ≈ {ui.tl(kayip)}."),
        "birlikte": (f"En sık birlikte alınan çift {cift[0]['a']} + {cift[0]['b']} ({cift[0]['n']} sipariş). Ortalama sepet "
                     f"{ui.sayi(S['basket'], 1)} ₺ ve {ui.sayi(S['b'], 1)} adet.") if cift else None,
        "gun": gun_cum or None,
        "ciro": (f"Seçili {et(a, b)} döneminde net ciro {ui.sayi(S['net'] / 1e6, 2)} M ₺, günlük {ui.sayi(S['dnet'] / 1000, 1)} bin ₺; "
                 f"{ui.sayi(S['ord'])} sipariş."),
    }
    chat.rafbot(bilgi)


# ======================================================================
# SAYFA: Pazarlama & Müşteri
# ======================================================================
def sayfa_pazarlama():
    tum = data.veri()
    vid = (st.session_state.get("dosya_adi", ""), len(tum))
    ui.baslik("Pazarlama & Müşteri", "Hangi müşteriye, hangi ürünle, ne zaman gitmeliyiz?")
    m, a, b, ca, cb = ui.filtreler(tum)
    K = w.fk()
    if m.empty:
        st.warning("Bu filtrelerle veri yok. Filtreleri gevşetip tekrar dene.")
        st.stop()
    et = ui.tarih_etiketi
    S, C = w.ozet_c(m, K, a, b), w.ozet_c(m, K, ca, cb)
    hN, hD = S["gun"] == C["gun"], S["gun"] != C["gun"]
    ch = an2.musteri_degisim(tum, m, a, b, ca, cb)
    mr = an.araliga_kes(m, a, b)
    perC = S["net"] / ch["aS"] if ch["aS"] else 0
    perCc = C["net"] / ch["aC"] if ch["aC"] else 0
    dpc = S["dnet"] / S["dcust"] if S["dcust"] else 0
    dpcc = C["dnet"] / C["dcust"] if C["dcust"] else 0

    st.markdown(w.ust_bilgi(S, C, a, b, ca, cb), unsafe_allow_html=True)
    w.kgrid([
        w.kart("Aktif müşteri", ui.sayi(ch["aS"]), w.chg(w.rel(ch["aS"], ch["aC"]), hN), "Günlük ort.", ui.sayi(S["dcust"]), w.chg(w.rel(S["dcust"], C["dcust"]), hD)),
        w.kart("Yeni müşteri", ui.sayi(ch["nw"]), w.chg(w.rel(ch["nw"], ch["nwC"]), hN), "Günlük ort.", ui.sayi(ch["nw"] / S["gun"]), w.chg(w.rel(ch["nw"] / S["gun"], ch["nwC"] / C["gun"]), hD)),
        w.kart("Satılan adet", ui.sayi(S["q"] / 1000, 1) + " bin", w.chg(w.rel(S["q"], C["q"]), hN), "Günlük ort.", ui.sayi(S["dq"]), w.chg(w.rel(S["dq"], C["dq"]), hD)),
        w.kart("Net ciro", ui.sayi(S["net"] / 1e6, 2) + " M ₺", w.chg(w.rel(S["net"], C["net"]), hN), "Günlük ort.", ui.sayi(S["dnet"] / 1000, 1) + " bin ₺", w.chg(w.rel(S["dnet"], C["dnet"]), hD)),
        w.kart("Müşteri başı ciro", ui.sayi(perC / 1000, 1) + " bin ₺", w.chg(w.rel(perC, perCc), hN), "Günlük ort.", ui.sayi(dpc) + " ₺", w.chg(w.rel(dpc, dpcc), hD), "#e0a526"),
    ])
    w.not_kutusu("Tüm paneller seçtiğin tarihlere göre hesaplanır. Yalnız 2. panel (aylar arası tekrar alım) seçtiğin yılın tamamını gösterir.")

    # ---------- 1. Müşteri tabanı değişimi ----------
    net = ch["aS"] - ch["aC"]
    with st.container(border=True):
        ui.bolum(1, "Müşteri tabanı bu dönemde nasıl değişti?", f"Karşılaştırma dönemindeki aktif müşteri {ui.sayi(ch['aC'])} → seçili dönemdeki aktif müşteri {ui.sayi(ch['aS'])}")
        w.vurgu(f"{ui.sayi(ch['nw'])} yeni ve {ui.sayi(ch['bk'])} geri dönen müşteri geldi, {ui.sayi(ch['ls'])} müşteri alım yapmadı. "
                f"Aktif müşteri {ui.sayi(abs(net))} {'arttı' if net >= 0 else 'azaldı'}.")
        k1, k2 = st.columns([3, 2])
        with k1:
            v = [-ch["ls"], ch["nw"], ch["bk"], net]
            ad = ["Kaybedilen", "Yeni", "Geri dönen", "Net değişim"]
            al = ["karşılaştırmada aldı, şimdi almadı", "ilk alımı bu dönemde", "daha önce almış, karşılaştırmada almamış", "aktif müşteri farkı"]
            mxa = max([abs(x) for x in v] + [1])
            fig = go.Figure(go.Bar(y=[f"{x}<br><span style='font-size:10px;color:#5b6475'>{y}</span>" for x, y in zip(ad, al)], x=v, orientation="h",
                                   marker_color=[ui.RENK["kirmizi"] if x < 0 else ui.RENK["yesil"] for x in v],
                                   text=[("+" if x >= 0 else "−") + ui.sayi(abs(x)) for x in v], textposition="outside", cliponaxis=False))
            fig.update_yaxes(autorange="reversed", automargin=True)
            fig.update_xaxes(zeroline=True, zerolinecolor="#1f2a44", zerolinewidth=2, showgrid=False, showticklabels=False,
                             range=[-mxa * 1.4, mxa * 1.4], title_text="← azalma  |  0  |  artış →")
            fig.update_layout(height=290, margin=dict(l=10, r=10, t=6, b=0), plot_bgcolor="white")
            ui.plot(fig)
        with k2:
            for t, x in (("Karşılaştırma dönemi · aktif müşteri", ch["aC"]), ("Seçili dönem · aktif müşteri", ch["aS"]),
                         ("İki dönemde de alan (devam eden)", ch["ct"])):
                st.markdown(f"<div class='out'><div class='t'>{t}</div><div class='v'>{ui.sayi(x)}</div></div><div style='height:6px'></div>", unsafe_allow_html=True)
            st.caption("Veri ilk gününde başladığı için ilk ayda “yeni” sayısı yüksek çıkar.")

    # ---------- 2. Kohort elde tutma ----------
    KS, KG = an2.kohort_ceyrek(m)
    with st.container(border=True):
        ui.bolum(2, "Yeni müşteriler bizimle kalıyor mu?", "İlk siparişini aynı çeyrekte veren müşteriler bir kohort. Sonraki çeyreklerde tekrar sipariş verenlerin yüzdesi")
        if KG[1] is None:
            st.info("Kohort hesabı için en az iki çeyreklik veri gerekir.")
        else:
            w.vurgu(f"İlk siparişten sonraki çeyrekte müşterilerin %{ui.sayi(KG[1], 0)}’i tekrar sipariş verdi"
                    + (f", ikinci çeyrekte %{ui.sayi(KG[2], 0)}’i." if KG[2] is not None else "."))
            o1, o2, o3 = st.columns(3)
            o1.markdown(f"<div class='out'><div class='t'>İlk siparişten sonraki çeyrekte sipariş verenler</div><div class='v'>%{ui.sayi(KG[1], 0)}</div></div>", unsafe_allow_html=True)
            o2.markdown(f"<div class='out'><div class='t'>İlk siparişten sonra 2. çeyrekte tekrar sipariş verenler</div><div class='v'>{'%' + ui.sayi(KG[2], 0) if KG[2] is not None else '–'}</div></div>", unsafe_allow_html=True)
            en = max((x for x in KS if x[2][0] is not None), key=lambda x: x[2][0])
            o3.markdown(f"<div class='out'><div class='t'>En iyi kohort (1. çeyrek sonrası)</div><div class='v'>{en[0]} · %{ui.sayi(en[2][0], 0)}</div></div>", unsafe_allow_html=True)
            tip = w.info("Satır: ilk siparişini o çeyrekte veren müşteriler. Sütun: bu müşterilerin ilk siparişten 1, 2 ve 3 çeyrek sonra, o çeyrekte en az bir sipariş verenlerin yüzdesi. Boş hücre: veri henüz o çeyreğe ulaşmıyor.")
            h = (f"<table class='t' style='margin-top:10px'><thead><tr><th>İlk sipariş çeyreği{tip}</th><th class='r'>İlk kez sipariş veren</th>"
                 "<th style='text-align:center'>Sonraki çeyrekte sipariş verenler</th><th style='text-align:center'>2. çeyrekte tekrar sipariş verenler</th>"
                 "<th style='text-align:center'>3. çeyrekte tekrar sipariş verenler</th></tr></thead><tbody>")
            for lab, n, yuz in KS:
                h += f"<tr><td style='font-weight:600'>{lab}</td><td class='r'>{ui.sayi(n)}</td>"
                for v in yuz:
                    if v is None:
                        h += "<td style='text-align:center;color:#9aa3b2'>–</td>"
                    else:
                        h += f"<td style='text-align:center;font-weight:600;background:{w.mix('#f6f3ee', '#2f8a73', min(1, v / 60))};color:{'#fff' if v > 38 else '#1f2a44'}'>%{ui.sayi(v, 0)}</td>"
                h += "</tr>"
            st.markdown(h + "</tbody></table>", unsafe_allow_html=True)
            st.caption("Veri 2 Ocak 2020’de başladığı için Ç1 kohortu, o tarihte zaten müşteri olanları da içerir; bu yüzden diğer kohortlardan yüksek çıkabilir.")

    # ---------- 3. RFM ----------
    rf, onceden = an2.rfm(tum, m, a, b)
    cnt_seg = [int((rf["seg"] == i).sum()) for i in range(6)] + [len(onceden)]
    net_seg = [rf.loc[rf["seg"] == i, "net"].sum() for i in range(6)]
    net_top = sum(net_seg) or 1
    harita = np.zeros((5, 5), dtype=int)
    for _, r in rf.iterrows():
        harita[5 - int(r["FM"]), int(r["R"]) - 1] += 1
    churn, auc = w.churn_c(tum, vid)
    with st.container(border=True):
        ui.bolum(3, "Müşteriler davranışlarına göre kimler?",
                 f"{et(a, b)} döneminde sipariş veren {ui.sayi(len(rf))} müşteri, son sipariş zamanı, sipariş sıklığı ve ciroya göre altı gruba ayrıldı (RFM)" +
                 w.info("Son sipariş (Recency), sipariş sıklığı (Frequency) ve ciro (Monetary) seçili dönemde müşterileri 1–5 arası puanlar. Puanlara göre altı grup oluşur."))
        s1, s2 = st.columns([1, 1.3])
        with s1:
            g = "<div style='display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px'>"
            for r in range(5):
                for c in range(5):
                    sg = an2.seg_of(c + 1, 5 - r)
                    g += (f"<div style='background:{an2.SEG_RENK[sg]};color:{an2.SEG_YAZI[sg]};border-radius:6px;min-height:46px;display:flex;flex-direction:column;"
                          f"align-items:center;justify-content:center;line-height:1.15' title='{an2.SEG_AD[sg]}: {harita[r][c]} müşteri'>"
                          f"<b style='font-size:13px'>{harita[r][c]}</b><span style='font-size:9px;opacity:.85'>müşteri</span></div>")
            g += "</div><div style='font-size:11px;color:#5b6475;text-align:center;margin-top:4px'>son sipariş: eski → yeni · satırlar: sipariş sıklığı ve ciro, yüksek üstte</div>" \
                 "<div style='font-size:11px;color:#5b6475;text-align:center'>Kutudaki sayı: müşteri adedi · renk: segment</div>"
            st.markdown(g, unsafe_allow_html=True)
        with s2:
            satir = []
            for i in range(7):
                nokta = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{an2.SEG_RENK[i]};margin-right:6px'></span>"
                satir.append([nokta + an2.SEG_AD[i] + w.info(an2.SEG_ACIKLAMA[i]), ui.sayi(cnt_seg[i]),
                              f"%{ui.sayi(cnt_seg[i] / max(1, len(rf)) * 100, 1)}" if i < 6 else "–",
                              f"%{ui.sayi(net_seg[i] / net_top * 100, 1)}" if i < 6 else "–"])
            satir.append(["<b>Tüm müşteriler</b>", f"<b>{ui.sayi(len(rf) + len(onceden))}</b>", "", ""])
            st.markdown(w.tablo(["Segment", "Müşteri adedi", "Müşteri payı", "Ciro payı"], satir, sag=(1, 2, 3)), unsafe_allow_html=True)
        sec = st.selectbox("Excel için segment", ["Tüm müşteriler"] + an2.SEG_AD[:6], key="rfm_sec")
        secim = rf if sec == "Tüm müşteriler" else rf[rf["seg"] == an2.SEG_AD.index(sec)]
        cr = an.araliga_kes(m, ca, cb).groupby("CustomerID").agg(cs=("InvoiceNo", "nunique"), cn=("net", "sum"))
        ilk_t = tum.groupby("CustomerID")["gun"].min()
        pc = an.pareto(mr, "CustomerID")
        il = tum.groupby("CustomerID")["City"].agg(lambda s: s.mode().iloc[0] if s.notna().any() and len(s.mode()) else "")
        ex = pd.DataFrame({
            "Müşteri": secim.index, "İl": il.reindex(secim.index).values,
            "Segment (RFM)": secim["seg"].map(lambda i: an2.SEG_AD[i]).values, "Segment (A-B-C)": pc["seg"].reindex(secim.index).values,
            "Seçili aralıkta sipariş sayısı": secim["sip"].values, "Seçili aralıkta net ciro (₺)": secim["net"].round(0).values,
            "Karşılaştırma aralığında sipariş sayısı": cr["cs"].reindex(secim.index).fillna(0).astype(int).values,
            "Karşılaştırma aralığında net ciro (₺)": cr["cn"].reindex(secim.index).fillna(0).round(0).values,
            "İlk sipariş": ilk_t.reindex(secim.index).dt.strftime("%d.%m.%Y").values,
            "Son sipariş": secim["son"].dt.strftime("%d.%m.%Y").values, "Kaç gündür sipariş yok": secim["rec"].values,
            "Bırakma riski (30 gün)": (churn["p"].reindex(secim.index) * 100).round(0).values})
        st.download_button(f"Excel’e indir: {sec} · {ui.sayi(len(ex))} müşteri", ui.xlsx(ex, "Müşteriler"),
                           file_name="musteriler_" + sec.lower().replace(" ", "_").replace("/", "") + ".xlsx", mime=w.XLSX, type="primary")

    # ---------- 4. Sipariş aralığı ----------
    gd = an2.gecikme(tum, rf, b)
    with st.container(border=True):
        ui.bolum(4, "Sipariş aralığı uzayan müşteriler kimler?", f"Her müşterinin kendi tipik sipariş aralığına göre, {et(b, b)} itibarıyla")
        rf2 = rf.assign(g=gd.reindex(rf.index))
        sayim = {i: [int(((rf2["seg"] == i) & (rf2["g"] == k)).sum()) for k in range(4)] for i in range(6)}
        ond = gd.reindex(onceden).value_counts()
        sayim[6] = [int(ond.get(k, 0)) for k in range(4)]
        cok_gec = sum(sayim[i][2] for i in range(7))
        vip = sayim[0][2] + sayim[1][2]
        w.vurgu(f"{ui.sayi(cok_gec)} müşteri, kendi tipik aralığının 2 katından uzun süredir sipariş vermedi" + (f"; {ui.sayi(vip)}’i Şampiyon veya Sadık." if vip else "."))
        ren = ["#3d9a84", "#e0a526", "#d0574f", "#cfc9bd"]
        ad = ["Zamanında", "Gecikmiş", "Çok gecikmiş", "Yeterli geçmiş yok"]
        tn = ["Son siparişten bu yana geçen süre, müşterinin tipik sipariş aralığını aşmadı. Tipik aralık: müşterinin sipariş günleri arasındaki ortanca gün sayısı.",
              "Geçen süre tipik aralığın 1 ile 2 katı arasında.", "Geçen süre tipik aralığın 2 katından fazla. Müşteri büyük olasılıkla alışverişi yavaşlattı veya bıraktı.",
              "Müşterinin 3’ten az sipariş günü var, tipik aralık hesaplanamadı."]
        st.markdown("<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-bottom:8px'>" + "".join(
            f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{ren[i]};margin-right:4px'></span>{ad[i]}{w.info(tn[i])}</span>" for i in range(4)) + "</div>", unsafe_allow_html=True)
        bars = ""
        for i in range(7):
            t = sum(sayim[i])
            if not t:
                continue
            bars += (f"<div style='display:grid;grid-template-columns:150px minmax(0,1fr) 40px;gap:8px;align-items:center;padding:4px 0;font-size:12px'><span>{an2.SEG_AD[i]}</span>"
                     "<div style='display:flex;height:20px;border-radius:5px;overflow:hidden;background:#f0ebe2'>" +
                     "".join(f"<div style='flex:{v} 1 0;background:{ren[k]}' title='{ad[k]}: {v} müşteri'></div>" for k, v in enumerate(sayim[i])) +
                     f"</div><span style='text-align:right;color:#5b6475'>{t}</span></div>")
        st.markdown(bars, unsafe_allow_html=True)
        st.caption("Sağdaki sayı: gruptaki müşteri adedi. Bırakma olasılığı için Satış sayfasına bak.")

    # ---------- 5-6. İl ve birlikte alınanlar ----------
    p5, p6 = st.columns([1.3, 1])
    ild = an2.il_dagilimi(tum, mr)
    with p5, st.container(border=True):
        ui.bolum(5, "Hangi ilden ne kadar satıyoruz?", "Seçili dönemde cironun il payı, aktif müşteri adedi ve sipariş sayısı")
        if ild.empty:
            st.info("Bu aralıkta sipariş yok.")
        else:
            top = ild.head(8)
            tot = ild["net"].sum()
            w.vurgu(f"{top.index[0]} cironun %{ui.sayi(top['net'].iloc[0] / tot * 100, 0)}’ini getiriyor; ilk {len(top)} il toplam %{ui.sayi(top['net'].sum() / tot * 100, 0)}.")
            mx = top["net"].iloc[0]
            h = ""
            for il_ad, r in top.iterrows():
                h += (f"<div style='display:grid;grid-template-columns:80px minmax(0,1fr) 50px 170px;gap:8px;align-items:center;padding:4px 0;font-size:12px'><span>{il_ad}</span>"
                      f"<div style='height:16px;background:#f0ebe2;border-radius:4px'><div style='width:{r['net'] / mx * 100:.1f}%;height:100%;background:#d9622b;border-radius:4px'></div></div>"
                      f"<span style='text-align:right;font-weight:600'>%{ui.sayi(r['net'] / tot * 100, 1)}</span>"
                      f"<span style='text-align:right;color:#5b6475;white-space:nowrap'>{int(r['musteri'])} müşteri · {ui.sayi(r['siparis'])} sipariş</span></div>")
            st.markdown(h, unsafe_allow_html=True)
            st.caption(f"Diğer iller %{ui.sayi(max(0, 100 - top['net'].sum() / tot * 100), 1)}. İl, müşterinin en çok sipariş verdiği ildir.")
    cift = w.birlikte_c(m, K, a, b)
    gor, gorulen = [], set()
    for c in cift:
        k = " ".join(c["a"].split()[:2]).lower()
        if k not in gorulen and len(gor) < 8:
            gorulen.add(k)
            gor.append(c)
    with p6, st.container(border=True):
        ui.bolum(6, "Hangi ürünler birlikte alınıyor?", "Seçili dönemde aynı siparişte en sık görülen ürün çiftleri" +
                 w.info("Ürün çiftleri, seçili dönemde en çok satan ürünler arasından seçilir ve aynı siparişteki sipariş sayısına göre sıralanır."))
        if not gor:
            st.info("Bu aralıkta yeterli birlikte alım yok.")
        else:
            w.vurgu("Çiftlerin çoğu aynı serinin kitapları: paket fiyatla birlikte teklif edilebilir.")
            mx = gor[0]["n"]
            st.markdown("".join(
                f"<div style='padding:5px 0;font-size:12px'><div style='display:flex;justify-content:space-between;gap:8px'><span><b>{c['a']}</b> + <b>{c['b']}</b></span>"
                f"<span style='white-space:nowrap;font-weight:600'>{c['n']} sipariş</span></div><div style='height:6px;background:#f0ebe2;border-radius:3px;margin-top:3px'>"
                f"<div style='width:{c['n'] / mx * 100:.0f}%;height:100%;background:#3d9a84;border-radius:3px'></div></div></div>" for c in gor), unsafe_allow_html=True)

    # ---------- 7. Kategori tercihi ----------
    with st.container(border=True):
        ui.bolum(7, "Hangi grup hangi kategoriyi daha çok alıyor?", f"{et(a, b)} · grubun kategori payı ÷ tüm müşterilerin kategori payı · 100 = ortalama" +
                 w.info("Örnek: 150 = bu grup, o kategoriyi diğer müşterilere göre 1,5 kat fazla alıyor. Yalnız cirosu en yüksek 6 kategori gösterildi."))
        cats, KM = an2.kategori_endeksi(mr, rf)
        if not cats:
            st.info("Bu aralıkta yeterli veri yok.")
        else:
            en = max(((v, i, j) for i, r in enumerate(KM) for j, v in enumerate(r) if cnt_seg[i] > 0), default=None)
            if en:
                w.vurgu(f"{an2.SEG_AD[en[1]]} grubunun {cats[en[2]]} alımı ortalamanın {ui.sayi(en[0] / 100, 1)} katı.")
            st.markdown(w.sapma_tablo(an2.SEG_AD[:6], cats, KM), unsafe_allow_html=True)

    # ---------- RafBot ----------
    bilgi = {
        "ciro": f"Seçili dönemde {ui.sayi(ch['aS'])} müşteri sipariş verdi: " + ", ".join(f"{an2.SEG_AD[i]} {cnt_seg[i]}" for i in range(6)) + ".",
        "segment": f"Seçili dönemde {ui.sayi(ch['aS'])} müşteri sipariş verdi: " + ", ".join(f"{an2.SEG_AD[i]} {cnt_seg[i]}" for i in range(6)) + ".",
        "churn": f"{ui.sayi(cok_gec)} müşteri kendi tipik sipariş aralığının 2 katından uzun süredir sipariş vermedi.",
        "kopru": f"Aktif müşteri {ui.sayi(ch['aC'])} → {ui.sayi(ch['aS'])}: {ui.sayi(ch['nw'])} yeni, {ui.sayi(ch['bk'])} geri dönen, {ui.sayi(ch['ls'])} kaybedilen.",
        "birlikte": (f"Seçili dönemde en sık birlikte alınan çift: {gor[0]['a']} + {gor[0]['b']} ({gor[0]['n']} sipariş).") if gor else None,
        "indirim": None, "gun": None,
    }
    chat.rafbot(bilgi)


# ======================================================================
# SAYFA: Tedarik Zinciri
# ======================================================================
def sayfa_tedarik():
    @st.cache_data(show_spinner="Ürün sınıfları hesaplanıyor…")
    def c_abc(_tum, vid):
        return an2.abc_xyz(_tum)


    @st.cache_data(show_spinner=False)
    def _c_hafta(_tum, vid, urun):
        wk, gunluk = an2.haftalik(_tum, urun)
        sonuc, en = an2.geri_test(wk)
        return wk, gunluk, [{k: v for k, v in r.items() if k != "f"} for r in sonuc], en


    def c_hafta(tum, vid, urun):
        """Önbellekten gelen sonuca (pickle edilemeyen) tahmin fonksiyonlarını geri ekler."""
        wk, gunluk, sonuc, en = _c_hafta(tum, vid, urun)
        return wk, gunluk, [dict(r, f=f) for r, (_, f) in zip(sonuc, an2.MODELLER)], en


    @st.cache_data(show_spinner=False)
    def c_satissiz(_tum, vid):
        return an2.satissiz(_tum)


    tum = data.veri()
    vid = (st.session_state.get("dosya_adi", ""), len(tum))
    ui.baslik("Tedarik Zinciri", "Ne kadar stok tutmalı, ne zaman sipariş vermeli?")
    m, a, b, ca, cb = ui.filtreler(tum)
    K = w.fk()
    if m.empty:
        st.warning("Bu filtrelerle veri yok. Filtreleri gevşetip tekrar dene.")
        st.stop()
    et = ui.tarih_etiketi
    S, C = w.ozet_c(m, K, a, b), w.ozet_c(m, K, ca, cb)
    hN, hD = S["gun"] == C["gun"], S["gun"] != C["gun"]
    AV = "#3b6fd1"

    st.markdown(w.ust_bilgi(S, C, a, b, ca, cb), unsafe_allow_html=True)
    w.kgrid([
        w.kart("Satılan adet", ui.sayi(S["q"] / 1000, 1) + " bin", w.chg(w.rel(S["q"], C["q"]), hN), "Günlük ort.", ui.sayi(S["dq"]) + " adet", w.chg(w.rel(S["dq"], C["dq"]), hD), AV),
        w.kart("Sipariş adedi", ui.sayi(S["ord"]), w.chg(w.rel(S["ord"], C["ord"]), hN), "Günlük ort.", ui.sayi(S["dord"], 1), w.chg(w.rel(S["dord"], C["dord"]), hD), AV),
        w.kart("Ort. sepet adedi", ui.sayi(S["b"]) + " adet", w.chg(w.rel(S["b"], C["b"]), hN), "Günlük ort.", ui.sayi(S["dbq"]) + " adet", w.chg(w.rel(S["dbq"], C["dbq"]), hD), AV),
        w.kart("Aktif müşteri", ui.sayi(S["musteri"]), w.chg(w.rel(S["musteri"], C["musteri"]), hN), "Günlük ort.", ui.sayi(S["dcust"]), w.chg(w.rel(S["dcust"], C["dcust"]), hD), AV),
        w.kart("Net ciro", ui.sayi(S["net"] / 1e6, 2) + " M ₺", w.chg(w.rel(S["net"], C["net"]), hN), "Günlük ort.", ui.sayi(S["dnet"] / 1000, 1) + " bin ₺", w.chg(w.rel(S["dnet"], C["dnet"]), hD), "#e0a526"),
    ])
    bit = tum["gun"].max()
    w.not_kutusu(f"Üstteki göstergeler seçtiğin tarihlere göre hesaplanır. Paneller verinin son gününe ({et(bit, bit)}) ve tüm veriye göre hazırlandı.")

    qadet, kum, abc, xyz = c_abc(tum, vid)
    nprod = len(qadet)
    cnt = {k: int((abc == k).sum()) for k in "ABC"}
    a_pay = cnt["A"] / nprod * 100

    # ---------- 1-2 ----------
    p1, p2 = st.columns([1.2, 1])
    with p1, st.container(border=True):
        ui.bolum(1, "Adedin %80’ini hangi ürünler getiriyor?", "Ürünler satılan adede göre sıralandı")
        w.vurgu(f"Ürünlerin %{ui.sayi(a_pay, 0)}’i ({ui.sayi(cnt['A'])} ürün) satılan adedin %80’ini oluşturuyor.")
        x = np.arange(1, nprod + 1) / nprod * 100
        ix = np.unique(np.linspace(0, nprod - 1, 300).astype(int))
        bx = (cnt["A"] + cnt["B"]) / nprod * 100
        fig = go.Figure()
        for x0, x1, rn in ((0, a_pay, "#d6ece5"), (a_pay, bx, "#fdf0cf")):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=rn, line_width=0, layer="below")
        fig.add_hline(y=80, line_dash="dot", line_color=ui.RENK["yesil"])
        fig.add_scatter(x=np.r_[0, x[ix]], y=np.r_[0, kum.values[ix] * 100], mode="lines", line=dict(color="#1f2a44", width=2.5),
                        hovertemplate="%{x:.1f}% ürün → %{y:.1f}% adet<extra></extra>")
        fig.add_annotation(x=a_pay, y=80, text=f"A: ürünlerin %{ui.sayi(a_pay, 1)}’i", showarrow=False, xanchor="left", yanchor="top", xshift=6, yshift=-6)
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=6, b=0), plot_bgcolor="white",
                          xaxis=dict(title="ürünler, adedi en yüksekten düşüğe sıralı (%)", range=[0, 100]),
                          yaxis=dict(title="birikimli adet payı (%)", range=[0, 102]))
        ui.plot(fig)
        st.markdown(f"<span class='pill g'>A · {ui.sayi(cnt['A'])} ürün · adetin %80’i</span><span class='pill y'>B · {ui.sayi(cnt['B'])} ürün · %15</span>"
                    f"<span class='pill'>C · {ui.sayi(cnt['C'])} ürün · %5</span>", unsafe_allow_html=True)
        cls = abc + xyz
        ust = qadet.head(10)
        katg = tum.dropna(subset=["Category"]).groupby("Description")["Category"].first()
        sinif_renk = {"A": "background:#14775f;color:#fff", "B": "background:#bfe3d9;color:#14513f", "C": "background:#eceff4;color:#3a4357"}
        satir = []
        for i, (u, v) in enumerate(ust.items(), 1):
            c = cls[u]
            satir.append([str(i), f"<b>{u if len(u) < 46 else u[:46] + '…'}</b><div style='font-size:11px;color:#5b6475'>{katg.get(u, '')}</div>",
                          f"<span style='display:inline-block;padding:2px 9px;border-radius:6px;font-size:12px;font-weight:700;{sinif_renk[c[0]]}'>{c}</span>",
                          f"<b>{ui.sayi(v)}</b>", f"%{ui.sayi(kum[u] * 100, 1)}"])
        st.markdown("<div style='font-weight:700;font-size:14px;margin:12px 0 4px'>En çok satan ürünler</div>" +
                    w.tablo(["#", "Ürün", "Sınıf", "Adet", "Kümülatif"], satir, sag=(3, 4)), unsafe_allow_html=True)
    cls = abc + xyz
    toplam_q = qadet.sum()
    hucre = {k: (int((cls == k).sum()), float(qadet[cls == k].sum() / toplam_q * 100)) for k in an2.POLITIKA}
    with p2, st.container(border=True):
        ui.bolum(2, "Hangi ürünün talebi düzensiz?", "Satış hacmi (A-B-C) ve aylık satışın oynaklığı (X-Y-Z)")
        w.vurgu(f"A ürünlerinin {ui.sayi(hucre['AZ'][0])} tanesi düzensiz: adedin %{ui.sayi(hucre['AZ'][1], 0)}’i stok sürprizi riski taşıyor.")
        mxu = max(v[1] for v in hucre.values()) or 1
        g = "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px'>"
        for k, (n, u) in hucre.items():
            t = min(1, u / mxu)
            g += (f"<div style='background:{w.mix('#e8f2ef', '#2f8a73', t ** 0.5)};color:{'#fff' if t > 0.45 else '#1f2a44'};border-radius:8px;padding:8px 10px;min-height:78px'>"
                  f"<div style='font-size:12px;font-weight:700'>{k}</div><div style='font-size:11px'>{ui.sayi(n)} ürün</div><div style='font-size:11px'>adetin %{ui.sayi(u, 1)}’i</div>"
                  f"<div style='font-size:11px;font-weight:600;margin-top:2px'>{an2.POLITIKA[k]}</div></div>")
        st.markdown(g + "</div>", unsafe_allow_html=True)
        st.caption("Düzenli: aylık satışın dalgalanması düşük. Dalgalı: orta. Düzensiz: çok yüksek (bir ay çok, bir ay hiç).")

    # ---------- 3. Talep tahmini ----------
    top12 = qadet.head(12).index.tolist()
    with st.container(border=True):
        ui.bolum(3, "Önümüzdeki 8 haftada ne kadar talep bekleniyor?", "Ürünü seç; üç basit yöntem son 12 haftada denendi, hatası en düşük olan seçildi")
        urun = st.selectbox("Ürün", top12, format_func=lambda s: s if len(s) < 50 else s[:50] + "…", key="ted_urun")
        wk, gunluk, sonuc, en = c_hafta(tum, vid, urun)
        best = sonuc[en]
        f8 = best["f"](wk, 8)
        sum8, last8 = sum(f8), float(wk[-8:].sum())
        w.vurgu(f"{urun if len(urun) < 50 else urun[:50] + '…'}: {ui.sayi(sum8)} adet, son 8 haftadan " +
                (f"%{ui.sayi((sum8 / max(1, last8) - 1) * 100, 0)} fazla." if sum8 >= last8 else f"%{ui.sayi((1 - sum8 / max(1, last8)) * 100, 0)} az."))
        cols = st.columns(3)
        for i, (kol, r) in enumerate(zip(cols, sonuc)):
            on = i == en
            kol.markdown(f"<div style='border:1px solid {'#3d9a84' if on else '#e4dfd6'};background:{'#eaf4f0' if on else '#fff'};border-radius:10px;padding:8px 10px'>"
                         f"<div style='font-size:12px'>{r['ad']}</div><div style='font-size:13px;font-weight:700'>Hata %{ui.sayi(r['wape'] * 100, 1)}</div>"
                         + ("<div style='font-size:11px;color:#14513f;font-weight:600'>✓ seçilen</div>" if on else "") + "</div>", unsafe_allow_html=True)
        hist = wk[-26:]
        n = len(hist)
        hh = np.arange(n)
        e = [1.28 * best["sd"] * np.sqrt(k + 1) for k in range(8)]
        up = [v + x for v, x in zip(f8, e)]
        lo = [max(0, v - x) for v, x in zip(f8, e)]
        fig = go.Figure()
        fig.add_vrect(x0=n - 12, x1=n - 1, fillcolor="#efeae0", line_width=0, layer="below", annotation_text="test dönemi", annotation_position="top")
        fig.add_scatter(x=np.r_[np.arange(n, n + 8), np.arange(n + 7, n - 1, -1)], y=np.r_[up, lo[::-1]], fill="toself",
                        fillcolor="rgba(159,207,194,.55)", line=dict(width=0), hoverinfo="skip", name="olası aralık")
        fig.add_scatter(x=hh, y=hist, mode="lines", line=dict(color="#1f2a44", width=2), name="gerçek haftalık adet")
        fig.add_scatter(x=np.arange(n - 12, n), y=best["fcs"], mode="lines", line=dict(color="#3d9a84", width=1.5, dash="dash"), name="test tahmini")
        fig.add_scatter(x=np.r_[n - 1, np.arange(n, n + 8)], y=np.r_[hist[-1], f8], mode="lines", line=dict(color="#3d9a84", width=2.5, dash="dash"), name="tahmin")
        fig.add_vline(x=n - 1, line_dash="dot", line_color="#1f2a44")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", legend=dict(orientation="h", y=-0.15),
                          xaxis=dict(showticklabels=False, title="26 hafta önce → son hafta → +8 hafta"), yaxis=dict(title="haftalık adet", rangemode="tozero"))
        ui.plot(fig)
        st.caption("Hata %: 4 haftalık toplam talepte, tahmin ile gerçek arasındaki farkların toplamı ÷ gerçek toplam (son 12 deneme). Düşük olan iyi. "
                   "Veri tek yıl olduğu için mevsimsel yöntemler denenmedi.")

    # ---------- 4. Güvenlik stoğu ----------
    with st.container(border=True):
        ui.bolum(4, "Ne kadar güvenlik stoğu tutmalı?", "Aynı ürün için tedarik süresini ve hizmet seviyesini değiştir")
        s1, s2 = st.columns([1, 1.2])
        with s1:
            lt = st.slider("Tedarik süresi (hafta)", 1, 6, 2, key="ted_lt")
            sl = st.slider("Hizmet seviyesi (%)", 85, 99, 95, key="ted_sl")
            st.caption("Hizmet seviyesi: talebin stoktan karşılanma olasılığı. Güvenlik stoğu = z × haftalık sapma × √tedarik süresi. "
                       "Sipariş noktası = haftalık talep × süre + güvenlik stoğu.")
        stk = an2.stok(wk, gunluk, lt, sl)
        with s2:
            c = st.columns(2)
            for kol, (t, v, alt) in zip(c * 2, [("Güvenlik stoğu", f"{ui.sayi(stk['ss'])} adet", ""), ("Sipariş noktası", f"{ui.sayi(stk['rop'])} adet", "stok bu seviyeye inince sipariş ver"),
                                                ("Haftalık talep", f"{ui.sayi(stk['mu'])} adet", "son 26 hafta ortalaması"),
                                                ("Stoksuz gün", f"{stk['out']} / {stk['n']}", "son 26 haftayı bu kuralla oynatınca")]):
                kol.markdown(f"<div class='out'><div class='t'>{t}</div><div class='v'>{v}</div><div class='t'>{alt}</div></div><div style='height:6px'></div>", unsafe_allow_html=True)
        fig = go.Figure()
        d = np.arange(stk["n"])
        fig.add_hrect(y0=0, y1=stk["ss"], fillcolor="#fdf0cf", line_width=0, layer="below", annotation_text="güvenlik stoğu", annotation_position="top left")
        for dd in stk["outd"]:
            fig.add_vrect(x0=dd - 0.5, x1=dd + 0.5, fillcolor="rgba(208,87,79,.45)", line_width=0, layer="below")
        fig.add_scatter(x=d, y=stk["ser"], mode="lines", line=dict(color="#3d9a84", width=2), name="stok")
        fig.add_hline(y=stk["rop"], line_dash="dash", line_color="#1f2a44", annotation_text="sipariş noktası", annotation_position="top right")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", showlegend=False,
                          xaxis=dict(showticklabels=False, title="26 hafta önce → son hafta"), yaxis=dict(title="stok (adet)", rangemode="tozero"))
        ui.plot(fig)
        st.caption("Oynatma varsayımı: sipariş miktarı 4 haftalık talep, tedarik süresi sabit. Stok verisi olmadığı için gerçek stok değil, satış geçmişi kullanıldı. "
                   "Kırmızı çizgiler stoksuz günleri gösterir.")

    # ---------- 5. Mevsimsellik ----------
    cats, SM, yil = an2.mevsimsellik(tum)
    with st.container(border=True):
        ui.bolum(5, "Hangi ay hangi kategori güçlü?", f"Kategorinin aylık satılan adedi ÷ kendi aylık ortalaması · 100 = ortalama · {yil}")
        if len(cats):
            bs = max(((v, i, j) for i, r in enumerate(SM) for j, v in enumerate(r)), default=(0, 0, 0))
            w.vurgu(f"{cats[bs[1]]} talebi {w.AY_AD[bs[2]]} ayında ortalamanın {ui.sayi(bs[0] / 100, 1)} katına çıkıyor.")
            st.markdown(w.sapma_tablo(cats, ui.AYLAR, SM), unsafe_allow_html=True)

    # ---------- 6. Satışsız ürünler ----------
    ds = c_satissiz(tum, vid)
    bk = [(0, 30), (31, 90), (91, 180), (181, 10000)]
    bs_ad = ["0–30 gün", "31–90 gün", "91–180 gün", "181+ gün"]
    bs_n = [int(((ds["gun"] >= x) & (ds["gun"] <= y)).sum()) for x, y in bk]
    bs_r = ["#3d9a84", "#8bbf7a", "#e0a526", "#d0574f"]
    idle120 = int((ds["gun"] >= 120).sum())
    with st.container(border=True):
        ui.bolum(6, "Hangi ürünler uzun süredir satılmıyor?", f"Ürünlerin son satıştan bu yana geçen gün sayısı · {et(bit, bit)} itibarıyla")
        st.markdown(f"<div style='padding:10px 12px;border-radius:10px;background:#fdf0cf;color:#5c4200;font-size:13px;font-weight:600;line-height:1.4'>Uyarı: {ui.sayi(idle120)} ürün 120 günden uzun süredir "
                    f"satılmadı (toplam {ui.sayi(len(ds))} üründen). Stokta varsa eritme kampanyası veya iade konuşulmalı.</div>", unsafe_allow_html=True)
        mx = max(bs_n) or 1
        st.markdown("".join(f"<div style='display:grid;grid-template-columns:80px minmax(0,1fr) 60px;gap:8px;align-items:center;padding:4px 0;font-size:12px'><span>{bs_ad[i]}</span>"
                            f"<div style='height:18px;background:#f0ebe2;border-radius:4px'><div style='width:{v / mx * 100:.1f}%;height:100%;background:{bs_r[i]};border-radius:4px'></div></div>"
                            f"<span style='text-align:right;font-weight:600'>{ui.sayi(v)}</span></div>" for i, v in enumerate(bs_n)), unsafe_allow_html=True)
        esik = st.selectbox("Liste", [120, 180, 270], format_func=lambda x: f"{x}+ gün", key="ted_idle")
        lst = ds[ds["gun"] >= esik].sort_values("gun", ascending=False)
        ex = pd.DataFrame({"Ürün": lst.index, "Kategori": lst["kategori"].values, "Kaç gündür satılmadı": lst["gun"].values,
                           "Toplam adet": lst["adet"].values, "Toplam net ciro (₺)": lst["net"].round(0).values})
        st.download_button(f"Excel’e indir: satışsız ürün listesi · {ui.sayi(len(ex))} ürün · {esik}+ gün", ui.xlsx(ex, "Satışsız ürünler"),
                           file_name=f"satissiz_urunler_{esik}_gun.xlsx", mime=w.XLSX, type="primary")

    # ---------- 7. Stok planı ----------
    with st.container(border=True):
        ui.bolum(7, "En çok satan 12 ürün için stok planı", f"Tedarik süresi {lt} hafta, hizmet seviyesi %{sl} ile önerilen stok · yukarıdaki ayarlarla değişir")
        satir, stil = [], []
        for u in top12:
            wk_u, gk_u, sn_u, en_u = c_hafta(tum, vid, u)
            f4 = sum(sn_u[en_u]["f"](wk_u, 4))
            k = an2.stok(wk_u, gk_u, lt, sl)
            hata = sn_u[en_u]["wape"]
            sinif = cls[u]
            satir.append([u if len(u) < 40 else u[:40] + "…", f"<span class='pill g'>{sinif}</span>", ui.sayi(f4),
                          f"<span style='color:{'#8f2a24' if hata > 0.2 else '#14513f'};font-weight:600'>%{ui.sayi(hata * 100, 1)}</span>",
                          ui.sayi(k["ss"]), ui.sayi(k["rop"]), an2.POLITIKA[sinif]])
            stil.append("background:#eaf4f0" if u == urun else "")
        st.markdown(w.tablo(["Ürün", "Sınıf", "4 haftalık talep", "Hata %", "Güvenlik stoğu", "Sipariş noktası", "İzlenecek politika"], satir, sag=(2, 3, 4, 5), stil=stil), unsafe_allow_html=True)
        st.caption("Hata %’si 20’nin üstündeyse (kırmızı) tahmine fazla güvenme, siparişe göre tedarik daha güvenli.")

    # ---------- RafBot ----------
    bilgi = {
        "ciro": f"Seçili {et(a, b)} döneminde {ui.sayi(S['q'])} adet satıldı, günlük {ui.sayi(S['dq'])} adet.",
        "indirim": f"{urun} için {lt} hafta tedarik süresi ve %{sl} hizmet seviyesiyle güvenlik stoğu ≈ {ui.sayi(stk['ss'])}, sipariş noktası ≈ {ui.sayi(stk['rop'])} adet.",
        "churn": f"{ui.sayi(idle120)} ürün 120+ gündür satılmadı; {ui.sayi(bs_n[3])} ürün 181+ gündür. Listeyi 6. panelden Excel’e alabilirsin.",
        "kopru": f"{urun} için en iyi yöntem “{best['ad']}” (hata %{ui.sayi(best['wape'] * 100, 1)}). Önümüzdeki 8 haftada ≈ {ui.sayi(sum8)} adet bekleniyor.",
        "segment": f"Ürünlerin %{ui.sayi(a_pay, 1)}’i adedin %80’ini getiriyor (A). A ürünlerinin {ui.sayi(hucre['AZ'][0])} tanesinde talep düzensiz.",
        "birlikte": None, "gun": None,
    }
    chat.rafbot(bilgi)


# ======================================================================
# SAYFA: Veri Kalitesi
# ======================================================================
def sayfa_kalite():
    tum = data.veri()
    kal = st.session_state.get("kalite") or {}
    ui.baslik("Veri Kalitesi", "Bu verilere ne kadar güvenebiliriz ve rakamlar nasıl hesaplandı?")
    ui.filtreler(tum)


    @st.cache_data(show_spinner="Norm aşımı hesaplanıyor…")
    def c_norm(_tum, vid):
        return an2.norm_asimi(_tum)


    vid = (st.session_state.get("dosya_adi", ""), len(tum))
    ham = kal.get("ham_satir", len(tum))
    tem = kal.get("temiz_satir", len(tum))
    atilan = kal.get("atilan", ham - tem)
    nsip = tum["InvoiceNo"].nunique() if "InvoiceNo" in tum.columns else len(tum)
    ilk, son = tum["gun"].min(), tum["gun"].max()
    st.markdown(f"<span class='pill g'>{ui.sayi(tem)} / {ui.sayi(ham)} satır kullanıldı · %{ui.sayi(tem / ham * 100, 3)}</span>"
                f"<span class='pill'>{ui.tarih_etiketi(ilk.date(), son.date())}</span>"
                f"<span class='pill'>{ui.sayi(nsip)} sipariş · {ui.sayi(tum['CustomerID'].nunique())} müşteri</span>", unsafe_allow_html=True)

    top, kn = c_norm(tum, vid)
    asan_say = top["asan_siparis"]
    fazla = top["fazla"]

    # 1
    with st.container(border=True):
        ui.bolum(1, "Veri ön işleme · 5 adım", "Ham dosyadan panellere giden yol")
        adimlar = [("1 · YÜKLEME", ui.sayi(ham), f"satır okundu · {kal.get('ham_sutun', '?')} sütun"),
                   ("2 · TEMİZLİK", "−" + ui.sayi(atilan), "sıfır/negatif satır çıkarıldı"),
                   ("3 · HESAPLAMA", "+" + str(kal.get("yeni_sutun", "?")), "yeni sütun: brüt ciro, net ciro, indirim"),
                   ("4 · SEGMENTLEME", "A · B · C", "müşteri ve ürün, ciro payına göre"),
                   ("5 · ANOMALİ DENETİMİ", ui.sayi(asan_say), "sipariş normdan fazla indirimli")]
        for kol, (t, v, s) in zip(st.columns(5), adimlar):
            kol.markdown(f"<div style='background:#faf7f2;border:1px solid #e4dfd6;border-radius:12px;padding:10px 12px;height:100%'><div style='font-size:10px;font-weight:700;color:#5b6475;letter-spacing:.04em'>{t}</div>"
                         f"<div style='font-family:Poppins;font-weight:700;font-size:20px;margin:4px 0'>{v}</div><div style='font-size:11px;color:#5b6475'>{s}</div></div>", unsafe_allow_html=True)

    # 2
    bos = kal.get("bos", {})
    ad = {"InvoiceNo": "Sipariş no", "CustomerID": "Müşteri no", "InvoiceDate": "Tarih", "Description": "Ürün adı", "Category": "Kategori", "City": "Şehir", "StockCode": "Stok kodu"}
    satir = []
    for k, e in ad.items():
        if k in bos:
            n = bos[k]
            satir.append((e, (1 - n / ham) * 100, f"{ui.sayi(n)} satır boş" if n else ""))
    if "StockCode" in bos and kal.get("bilimsel"):
        satir[-1] = ("Stok kodu", satir[-1][1], f"dolu ama {ui.sayi(kal['bilimsel'])} kod bilimsel gösterimde bozulmuş")
    with st.container(border=True):
        ui.bolum(2, "Alan doluluğu", "Her alanın yüzde kaçı dolu · sarı ve kırmızı olanlar dikkat ister")
        h = ""
        for e, p, n in satir:
            renk, pill = ("#3d9a84", "g") if p >= 99.995 else (("#e0a526", "y") if p >= 95 else ("#c25b50", "r"))
            if "bozulmuş" in n:
                renk, pill = "#c25b50", "r"
            elif n and p >= 99.9:
                renk, pill = "#3d9a84", "g"
            pc = ui.sayi(p, 2)
            pc = pc[:-3] if pc.endswith(",00") else pc
            h += (f"<div style='display:grid;grid-template-columns:minmax(150px,.7fr) 2fr 70px;gap:14px;align-items:center;padding:6px 0'><span>{e}</span>"
                  f"<div><div style='background:#ece7dd;border-radius:999px;height:16px'><div style='width:{p:.2f}%;background:{renk};height:16px;border-radius:999px'></div></div>"
                  + (f"<span class='pill {pill}' style='margin-top:5px;font-weight:600'>{n}</span>" if n else "") +
                  f"</div><b style='text-align:right'>%{pc}</b></div>")
        st.markdown(h, unsafe_allow_html=True)

    # 3
    tekrar = kal.get("tekrar", 0)
    with st.container(border=True):
        ui.bolum(3, "Tekrar eden satırlar")
        st.markdown(f"<div style='font-size:13px'>Birebir aynı satır <b>{ui.sayi(tekrar)}</b> <span class='pill y'>%{ui.sayi(tekrar / ham * 100, 1)}</span></div>"
                    f"<div style='font-size:12px;color:#5b6475;margin:4px 0 8px'>Silinmedi: aynı ürün aynı siparişte iki kez okutulmuş olabilir. Silmek cirodan yaklaşık %{ui.sayi(tekrar / ham * 100, 1)} götürürdü.</div>"
                    "<span class='pill'>Bilinen risk · küçük</span>", unsafe_allow_html=True)

    # 4
    with st.container(border=True):
        ui.bolum(4, "Hangi kategoride indirim normu en çok aşılıyor?", "Her kategorinin normal indirimi (norm) var. Normun 10 puan üstünde indirimle kesilen siparişlerin payı")
        st.markdown(f"<span class='pill r'>Her 100 siparişin {ui.sayi(asan_say / top['toplam'] * 100, 0)}'si normun üstünde indirimli</span>"
                    f"<span class='pill y'>Fazlası toplam ≈ {ui.tl(top['fazla'])}</span>", unsafe_allow_html=True)
        h = ""
        mx = max(float(kn["pay"].max()), 0.01)
        for kat, r in kn.head(7).iterrows():
            esik = ui.sayi((r["md"] + 0.10) * 100, 0)
            h += (f"<div style='display:grid;grid-template-columns:minmax(220px,1.3fr) 1.4fr minmax(130px,.7fr);gap:14px;align-items:center;padding:8px 0;border-bottom:1px solid #eee9df'>"
                  f"<div><div style='font-weight:700;font-size:14px'>{kat}</div>"
                  f"<div style='font-size:12px;color:#5b6475;line-height:1.35'>Siparişlerin %{ui.sayi(r['pay'] * 100, 0)}'inde ({ui.sayi(r['asan'])} sipariş) indirim %{esik} ve üzeri · norm %{ui.sayi(r['md'] * 100, 0)}</div></div>"
                  f"<div style='background:#ece7dd;border-radius:999px;height:14px'><div style='width:{r['pay'] / mx * 90:.1f}%;background:#c25b50;height:14px;border-radius:999px'></div></div>"
                  f"<div style='text-align:right'><div style='font-weight:700;font-size:14px'>%{ui.sayi(r['pay'] * 100, 0)} ({ui.sayi(r['asan'])} sipariş)</div>"
                  f"<div style='font-size:12px;color:#5b6475'>≈ {ui.tl(r['fazla'])} fazla</div></div></div>")
        st.markdown(h, unsafe_allow_html=True)

    # 5
    with st.container(border=True):
        ui.bolum(5, "Varsayımlar ve sınırlar", "Kararlarda bu dört noktaya dikkat edilmeli")
        kt = [("Varsayım", "Maliyet verisi yok", "Alış maliyeti ürüne göre değişir. Tek ürün planında maliyeti sen girersin; başlangıç değeri liste fiyatın %44'ü", "Satış · indirim planı"),
              ("Varsayım", "Stok verisi yok", "Güvenlik stoğu ve sipariş noktası talebe göre hesaplanıyor", "Tedarik · yönetici özeti"),
              ("Sınır", "Saat bilgisi yok", "Yalnız gün bazlı analiz yapılabiliyor", "Satış · haftanın günü"),
              ("Sınır", "Yalnız 2020", "Yıllık mevsimsellik öğrenilemiyor; tahmin hatası ≈ %25", "Yönetici · tahminler")]
        for kol, (t, b, s, y) in zip(st.columns(4), kt):
            kol.markdown(f"<div style='background:#faf7f2;border:1px solid #e4dfd6;border-radius:12px;padding:10px 12px;height:100%'><span class='pill {'y' if t == 'Sınır' else 'g'}'>{t}</span>"
                         f"<div style='font-weight:700;margin:4px 0'>{b}</div><div style='font-size:12px;line-height:1.5'>{s}</div>"
                         f"<div style='font-size:11px;color:#5b6475;margin-top:6px'>{y}</div></div>", unsafe_allow_html=True)

    # 6
    SOZLUK = {
        "Net ciro": ("Net ciro = Σ miktar × net fiyat", "İndirim düşüldükten sonra tahsil edilen tutar. Liste fiyat üzerinden hesaplanan brüt ciro ayrıca tutulur.", "Satış · KPI · Yönetici · karne"),
        "İndirim oranı": ("İndirim oranı = 1 − net ciro / brüt ciro", "Liste fiyata göre verilen toplam indirim. Dönem içinde ciro ağırlıklı hesaplanır.", "Satış · Yönetici"),
        "Fazla indirim": ("Fazla indirim = Σ (sipariş indirimi − kategori normu) × brüt tutar", "Yalnız normdan 10 puan fazla indirimli siparişler sayılır. Kategori normu, o kategorinin ortanca indirimidir.", "Satış · şüpheli sipariş · Veri kalitesi"),
        "Segment A-B-C": ("A: cironun ilk %80’i · B: sonraki %15 · C: son %5", "Müşteri ve ürünler ciroya göre büyükten küçüğe sıralanır, kümülatif paya göre bölünür.", "Satış · Yönetici · Tedarik"),
        "Churn riski": ("Risk = P(30 gün içinde sipariş vermeme)", "Geçmiş sipariş sıklığı, son siparişten geçen süre ve tutarlardan öğrenen bir model hesaplar. %70 ve üzeri yüksek risk.", "Satış · Pazarlama · Yönetici"),
        "Sipariş noktası": ("Sipariş noktası = ort. günlük talep × temin süresi + güvenlik stoğu", "Güvenlik stoğu = z × talep sapması × √temin süresi. Stok seviyesi bu noktaya inince sipariş verilir.", "Tedarik"),
        "Talep tahmini": ("Üç yöntemden 4 haftalık hatası en küçük olan seçilir", "Son 4 hafta ortalaması, son 8 hafta ortalaması ve sönümlü trend. Hata, geçmişte denenerek ölçülür.", "Tedarik · Yönetici"),
    }
    with st.container(border=True):
        ui.bolum(6, "Metrik sözlüğü", "Bir metriğe bas: formülü ve hangi sayfada kullanıldığını gör")
        sec = st.radio("Metrik", list(SOZLUK), horizontal=True, label_visibility="collapsed", key="kal_metrik")
        f, a, y = SOZLUK[sec]
        st.markdown(f"<div style='background:#faf7f2;border-radius:10px;padding:12px 14px'><div style='font-family:Poppins;font-weight:700'>{f}</div>"
                    f"<div style='font-size:12px;margin:4px 0'>{a}</div><span class='pill'>{y}</span></div>", unsafe_allow_html=True)

    # 7
    with st.container(border=True):
        ui.bolum(7, "İşlenmiş veriden örnek 5 satır")
        gos = [c for c in ["InvoiceNo", "gun", "CustomerID", "Description", "Category", "City", "Quantity", "UnitPrice", "gross", "net", "disc"] if c in tum.columns]
        st.dataframe(tum[gos].head(5), hide_index=True, use_container_width=True)

    bilgi = {"ciro": f"{ui.sayi(tem)} satırın tamamı kullanıldı (ham {ui.sayi(ham)}, atılan {ui.sayi(atilan)}).",
             "indirim": f"{ui.sayi(asan_say)} sipariş kategori normunun 10 puan üstünde indirimli; fazlası ≈ {ui.tl(fazla)}.",
             "kopru": None, "churn": None, "segment": None, "birlikte": None, "gun": None}
    chat.rafbot(bilgi)


yukle = st.Page(sayfa_yukle, title="Veri yükle", icon="📂", url_path="yukle", default=("df" not in st.session_state))
yonetici = st.Page(sayfa_yonetici, title="Yönetici Özeti", icon="🧭", url_path="yonetici", default=("df" in st.session_state))
satis = st.Page(sayfa_satis, title="Satış & Fiyat", icon="💰", url_path="satis")
pazarlama = st.Page(sayfa_pazarlama, title="Pazarlama & Müşteri", icon="🎯", url_path="pazarlama")
tedarik = st.Page(sayfa_tedarik, title="Tedarik Zinciri", icon="📦", url_path="tedarik")
kalite = st.Page(sayfa_kalite, title="Veri Kalitesi", icon="🔍", url_path="kalite")

st.session_state["_sayfa_nesneleri"] = {
    "yukle": yukle,
    "yonetici": yonetici,
    "satis": satis,
    "pazarlama": pazarlama,
    "tedarik": tedarik,
    "kalite": kalite,
}

pg = st.navigation([yonetici, satis, pazarlama, tedarik, kalite, yukle], position="hidden")
pg.run()
