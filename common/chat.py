"""RafRadar gelişmiş RafBot (Veriye Sor) yüzen sohbet asistanı modülü."""
from __future__ import annotations

import re
import pandas as pd
import streamlit as st
from common import analytics as an
from common import ui


def _sayfa_tespit(bilgi: dict) -> str:
    """Bilgi sözlüğündeki ipuçlarından aktif sayfayı tahmin eder."""
    c = str(bilgi.get("ciro") or "")
    k = str(bilgi.get("kopru") or "")
    ind = str(bilgi.get("indirim") or "")
    if "baz tahmin" in c or "aksiyon" in ind:
        return "yonetici"
    if "en iyi yöntem" in k or "tedarik süresi" in ind:
        return "tedarik"
    if "geri dönen" in k or "Şampiyonlar" in c:
        return "pazarlama"
    if "ham " in c or "temizlendi" in c or "atılan" in c:
        return "kalite"
    return "satis"


HIZLI_SORULAR = {
    "yonetici": [
        "Gelecek ay ne olur?",
        "En çok getiren aksiyon",
        "İlk kimi aramalı?",
        "Ciro neden değişti?",
        "En büyük 5 müşteri kim?",
    ],
    "satis": [
        "Ciro neden düştü?",
        "İlk kimi aramalı?",
        "Fazla indirim ne kadar?",
        "Birlikte alınanlar",
        "Pazartesi yoğunluğu",
    ],
    "pazarlama": [
        "Kaç müşteri kaybettik?",
        "Yeni müşteri kalıyor mu?",
        "Geciken müşteriler",
        "En çok satan il hangisi?",
        "RFM segmentleri",
    ],
    "tedarik": [
        "8 hafta talep tahmini",
        "Güvenlik stoğu ne kadar?",
        "Satışsız ürünler",
        "Düzensiz ürünler (AZ)",
        "Hangi ay hangi kategori güçlü?",
    ],
    "kalite": [
        "Kaç satır temizlendi?",
        "Fazla indirim ne kadar?",
        "Maliyet verisi var mı?",
        "Tekrar eden satırlar",
        "Metrik formülleri",
    ],
}


def _tr_norm(s: str) -> str:
    """Türkçe karakter duyarlı küçük harf dönüşümü (İ -> i, I -> ı)."""
    return (
        str(s)
        .replace("İ", "i")
        .replace("I", "ı")
        .lower()
        .replace("i\u0307", "i")
    )


def _canli_veri_ara(soru: str, tum: pd.DataFrame) -> str | None:
    """Kullanıcının sorusunda geçen müşteri kodu, il, kategori veya ürün adını canlı veride arar."""
    if tum is None or tum.empty:
        return None

    q_raw = soru.strip()
    q_low = _tr_norm(q_raw)

    # 1. Müşteri kodu sorgusu (örn. S00172, A00023, C00105 vb.)
    m_kod = re.search(r"\b([A-Za-z]\d{4,6})\b", q_raw)
    if m_kod:
        cid = m_kod.group(1).upper()
        sub = tum[tum["CustomerID"].str.upper() == cid]
        if not sub.empty:
            net = sub["net"].sum()
            sip = sub["InvoiceNo"].nunique()
            adet = sub["Quantity"].sum()
            il = sub["City"].iloc[0]
            son = sub["gun"].max()
            pasif = (tum["gun"].max() - son).days
            ort_ind = (1.0 - net / max(1.0, sub["gross"].sum())) * 100.0
            en_kat = sub.groupby("Category")["net"].sum().idxmax() if sub["Category"].notna().any() else "—"
            return (
                f"<b>{cid}</b> ({il}) müşteri özeti:<br>"
                f"• Toplam net ciro: <b>{ui.tl(net)}</b> ({ui.sayi(sip)} sipariş · {ui.sayi(adet)} adet)<br>"
                f"• Ortalama indirim: <b>%{ui.sayi(ort_ind, 1)}</b> · En çok aldığı kategori: <b>{en_kat}</b><br>"
                f"• Son sipariş: <b>{son:%d.%m.%Y}</b> ({pasif} gün önce)."
            )

    # 2. En büyük müşteriler / en çok alan müşteriler sorgusu
    if re.search(r"en büyük.*müşteri|en çok.*müşteri|top.*müşteri|ilk 5 müşteri", q_low):
        top5 = tum.groupby(["CustomerID", "City"]).agg(net=("net", "sum"), sip=("InvoiceNo", "nunique")).nlargest(5, "net")
        satirlar = [f"{i}. <b>{cid}</b> ({il}): {ui.tl(r['net'])} ({ui.sayi(r['sip'])} sipariş)" for i, ((cid, il), r) in enumerate(top5.iterrows(), 1)]
        return "Verideki en yüksek cirolu 5 müşteri:<br>" + "<br>".join(satirlar)

    # 3. En çok satan ürünler sorgusu
    if re.search(r"en çok satan ürün|best seller|en popüler kitap|ilk 5 ürün", q_low):
        top5 = tum.groupby("Description").agg(adet=("Quantity", "sum"), net=("net", "sum")).nlargest(5, "adet")
        satirlar = [f"{i}. <b>{u[:42]}</b>: {ui.sayi(r['adet'])} adet ({ui.tl(r['net'])})" for i, (u, r) in enumerate(top5.iterrows(), 1)]
        return "Adet bazında en çok satan 5 ürün:<br>" + "<br>".join(satirlar)

    # 4.En çok satan il sorgusu
    if re.search(r"en çok satan il|hangi il|iller|şehirler", q_low):
        top5 = tum.groupby("City").agg(net=("net", "sum"), mus=("CustomerID", "nunique"), sip=("InvoiceNo", "nunique")).nlargest(5, "net")
        tot_net = max(1.0, tum["net"].sum())
        satirlar = [
            f"{i}. <b>{il}</b>: {ui.tl(r['net'])} (%{ui.sayi(r['net'] / tot_net * 100, 1)}) · {ui.sayi(r['mus'])} müşteri"
            for i, (il, r) in enumerate(top5.iterrows(), 1)
        ]
        return "Ciroya göre ilk 5 il:<br>" + "<br>".join(satirlar)

    # 5. Şehir / İl adı sorgusu
    if "City" in tum.columns:
        iller = tum["City"].dropna().unique().tolist()
        for il in iller:
            il_low = _tr_norm(il)
            if len(il_low) >= 3 and re.search(rf"\b{re.escape(il_low)}", q_low):
                sub = tum[tum["City"] == il]
                net = sub["net"].sum()
                pay = net / max(1.0, tum["net"].sum()) * 100.0
                mus = sub["CustomerID"].nunique()
                sip = sub["InvoiceNo"].nunique()
                en_kat = sub.groupby("Category")["net"].sum().idxmax() if sub["Category"].notna().any() else "—"
                return (
                    f"<b>{il}</b> ili canlı veri özeti:<br>"
                    f"• Net ciro: <b>{ui.tl(net)}</b> (tüm cironun <b>%{ui.sayi(pay, 1)}</b>'i)<br>"
                    f"• Aktif müşteri: <b>{ui.sayi(mus)}</b> · Sipariş sayısı: <b>{ui.sayi(sip)}</b><br>"
                    f"• En güçlü kategori: <b>{en_kat}</b>."
                )

    # 6. Kategori adı sorgusu
    if "Category" in tum.columns:
        kategoriler = sorted(tum["Category"].dropna().unique().tolist(), key=len, reverse=True)
        for kat in kategoriler:
            kat_low = _tr_norm(kat)
            kelimeler = [w for w in re.split(r"[\s/&,-]+", kat_low) if len(w) >= 4]
            if kat_low in q_low or any(re.search(rf"\b{re.escape(w)}", q_low) for w in kelimeler):
                sub = tum[tum["Category"] == kat]
                net = sub["net"].sum()
                pay = net / max(1.0, tum["net"].sum()) * 100.0
                adet = sub["Quantity"].sum()
                ind = (1.0 - net / max(1.0, sub["gross"].sum())) * 100.0
                en_urun = sub.groupby("Description")["Quantity"].sum().idxmax()
                return (
                    f"<b>{kat}</b> kategorisi canlı veri özeti:<br>"
                    f"• Net ciro: <b>{ui.tl(net)}</b> (toplamın <b>%{ui.sayi(pay, 1)}</b>'i) · Satılan: <b>{ui.sayi(adet)} adet</b><br>"
                    f"• Ortalama indirim: <b>%{ui.sayi(ind, 1)}</b><br>"
                    f"• En çok satan ürünü: <b>{en_urun[:48]}</b>."
                )

    # 7. Ürün / Kitap adı arama (en az 4 harfli anlamlı kelime eşleşmesi)
    stop_words = {
        "nedir", "nasıl", "kadar", "hangi", "neden", "niçin", "kimler", "kime", "ürün", "ürünler",
        "müşteri", "müşteriler", "sipariş", "ciro", "indirim", "stok", "tahmin", "hafta", "günlük",
        "aylık", "yıllık", "satış", "fiyat", "pazarlama", "tedarik", "kalite", "segment", "analiz",
    }
    arama_kelimeleri = [w for w in re.findall(r"\w+", q_low) if len(w) >= 4 and w not in stop_words]
    if arama_kelimeleri:
        descs = tum["Description"].dropna().unique()
        eslesen = [d for d in descs if all(k in _tr_norm(d) for k in arama_kelimeleri)]
        if not eslesen and len(arama_kelimeleri) >= 1:
            eslesen = [d for d in descs if arama_kelimeleri[0] in _tr_norm(d)]
        if 1 <= len(eslesen) <= 15:
            sec_u = eslesen[0]
            sub = tum[tum["Description"] == sec_u]
            net = sub["net"].sum()
            adet = sub["Quantity"].sum()
            sip = sub["InvoiceNo"].nunique()
            liste_f = sub["UnitPrice"].median()
            ind = (1.0 - net / max(1.0, sub["gross"].sum())) * 100.0
            son = sub["gun"].max()
            ek = f" (ve benzer {len(eslesen) - 1} ürün)" if len(eslesen) > 1 else ""
            return (
                f"<b>{sec_u}</b>{ek} ürün özeti:<br>"
                f"• Satılan adet: <b>{ui.sayi(adet)}</b> ({ui.sayi(sip)} sipariş) · Net ciro: <b>{ui.tl(net)}</b><br>"
                f"• Liste fiyatı: <b>{ui.sayi(liste_f, 2)} ₺</b> · Ortalama indirim: <b>%{ui.sayi(ind, 1)}</b><br>"
                f"• Son satış tarihi: <b>{son:%d.%m.%Y}</b>."
            )

    return None


def _cevap_uret(soru: str, bilgi: dict, sayfa: str) -> str:
    """Soruya sayfa bağlamı, bilgi tabanı ve canlı veri üzerinden yanıt üretir."""
    s = soru.strip().lower()
    tum = st.session_state.get("df")

    # 1. Sayfaya özel hızlı eşleşmeler
    if sayfa == "yonetici":
        if re.search(r"gelecek ay|30 gün|tahmin|ne olur", s) and bilgi.get("ciro"):
            return bilgi["ciro"]
        if re.search(r"aksiyon|en çok getiren|simülasyon|getiri", s) and bilgi.get("indirim"):
            return bilgi["indirim"]
    elif sayfa == "pazarlama":
        if re.search(r"kaybet|yeni müşteri.*kaç|geri dönen|değişti|taban", s) and bilgi.get("kopru"):
            return bilgi["kopru"]
        if re.search(r"kalıyor mu|kohort|tekrar sipariş|tutma", s):
            return "Yeni müşterilerin ilk siparişten sonraki çeyrekte yaklaşık <b>%44'ü</b>, ikinci çeyrekte <b>%53'ü</b> tekrar sipariş veriyor; yani yeni müşterilerin yarısından fazlası ilk çeyrekten sonra sessizleşiyor. İlk 30–45 günde çapraz kategori kuponu önerilir."
        if re.search(r"gecik|aralık|uzayan", s) and bilgi.get("churn"):
            return bilgi["churn"]
        if re.search(r"rfm|davranış|şampiyon", s) and bilgi.get("segment"):
            return bilgi["segment"]
    elif sayfa == "tedarik":
        if re.search(r"8 hafta|talep|tahmin|holt", s) and bilgi.get("kopru"):
            return bilgi["kopru"]
        if re.search(r"güvenlik stoğu|sipariş noktası|stok", s) and bilgi.get("indirim"):
            return bilgi["indirim"]
        if re.search(r"satışsız|satılmayan|120|hareketsiz|yavaş", s) and bilgi.get("churn"):
            return bilgi["churn"]
        if re.search(r"düzensiz|xyz|az\b", s) and bilgi.get("segment"):
            return bilgi["segment"]

    # 2. Genel bilgi sözlüğü eşleşmeleri
    if re.search(r"neden düştü|neden değişti|köprü|fark|düşüş|artış", s) and bilgi.get("kopru"):
        return bilgi["kopru"]
    if re.search(r"kimi aramalı|churn|risk|bırak|terk|kayıp", s) and bilgi.get("churn"):
        return bilgi["churn"]
    if re.search(r"fazla indirim|şüpheli|anomali|norm|indirim", s) and bilgi.get("indirim"):
        return bilgi["indirim"]
    if re.search(r"birlikte|sepet|çift|seri", s) and bilgi.get("birlikte"):
        return bilgi["birlikte"]
    if re.search(r"pazartesi|haftanın günü|hafta sonu|yoğun", s) and bilgi.get("gun"):
        return bilgi["gun"]
    if re.search(r"pareto|segment|a-b-c|80", s) and bilgi.get("segment"):
        return bilgi["segment"]
    if re.search(r"kaç satır|temiz|veri kalitesi|bozulmuş|tekrar eden", s):
        kal = st.session_state.get("kalite") or {}
        ham = kal.get("ham_satir", 664154)
        tem = kal.get("temiz_satir", 664133)
        at = kal.get("atilan", ham - tem)
        tek = kal.get("tekrar", 6217)
        return (
            f"Ham verideki <b>{ui.sayi(ham)}</b> satırdan sıfır/negatif miktar veya fiyatlı <b>{ui.sayi(at)}</b> satır temizlendi; "
            f"<b>{ui.sayi(tem)}</b> satır (%99,997) kullanıldı. Birebir tekrar eden <b>{ui.sayi(tek)}</b> satır (%0,9) aynı siparişte çift okutma olabileceği için silinmedi."
        )
    if re.search(r"maliyet|kâr|brüt", s):
        return (
            "Veride ürün alış maliyeti sütunu bulunmuyor. Satış & Fiyat sayfasındaki <b>İndirim planı</b> modülünde alış maliyetini "
            "liste fiyatın yüzdesi olarak (varsayılan <b>%44</b>) kendin değiştirebilirsin."
        )
    if re.search(r"formül|metrik|sözlük|nasıl hesap", s):
        return (
            "<b>Temel Formüller:</b><br>"
            "• <b>Net ciro</b> = Σ (Miktar × Net Fiyat)<br>"
            "• <b>İndirim oranı</b> = 1 − (Net Ciro / Brüt Ciro)<br>"
            "• <b>Fazla indirim</b> = Σ (Sipariş İndirimi − Kategori Medyanı) × Brüt Tutar (10+ puan aşanlar)<br>"
            "• <b>Sipariş noktası</b> = Haftalık Talep × Tedarik Süresi + z × σ × √Süre."
        )

    # 3. Canlı DataFrame üzerinde varlık (müşteri, il, kategori, ürün) araması
    canli = _canli_veri_ara(soru, tum)
    if canli:
        return canli

    # 4. Ciro genel sorusu
    if re.search(r"ciro|satış|sipariş|durum|özet", s) and bilgi.get("ciro"):
        return bilgi["ciro"]

    # 5. Varsayılan yardımcı yanıt
    return (
        "Bu soruya özel doğrudan bir eşleşme bulamadım ancak şunları sorabilirsin:<br>"
        "• <b>Konular:</b> “Ciro neden düştü?”, “İlk kimi aramalı?”, “Fazla indirim ne kadar?”, “Birlikte alınanlar”, “8 hafta talep tahmini”, “Güvenlik stoğu”<br>"
        "• <b>Canlı veri sorgusu:</b> Bir müşteri kodu (örn. <code>S00172</code>), şehir adı (örn. <code>İstanbul</code>, <code>Ankara</code>), kategori (örn. <code>Polisiye</code>, <code>Sözlük</code>) veya kitap adı (örn. <code>Kral Şakir</code>, <code>Karantina</code>) yazabilirsin!"
    )


def rafbot(bilgi: dict):
    """Sağ altta yüzen (floating) Ayraç AI · Veriye sor sohbet kutusunu oluşturur."""
    sayfa = _sayfa_tespit(bilgi)
    ss = st.session_state
    ss.setdefault(
        "rafbot_msgs",
        [
            {
                "role": "assistant",
                "text": "Merhaba! Ben <b>Ayraç AI</b> 🔖 — RafRadar'ın akıllı kitap &amp; kırtasiye karar asistanıyım. Bu sayfadaki veriler, müşteri kodları (örn. <code>S00172</code>), şehirler, kategoriler veya ürünler hakkında bana soru sorabilir ya da hazır başlıklara tıklayabilirsin.",
            }
        ],
    )

    # Yüzen sağ-alt popover için özel CSS (tüm Streamlit sürümleriyle uyumlu)
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] {
            position: fixed !important;
            right: 22px !important;
            bottom: 20px !important;
            z-index: 999990 !important;
            width: auto !important;
        }
        div[data-testid="stPopover"] > div > button,
        div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
            background: #3d9a84 !important;
            color: #ffffff !important;
            border: 2px solid #ffffff !important;
            border-radius: 999px !important;
            padding: 10px 18px !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 8px 24px rgba(31, 42, 68, 0.28) !important;
            transition: transform 0.15s ease, background 0.15s ease !important;
        }
        div[data-testid="stPopover"] > div > button:hover,
        div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:hover {
            background: #2f7f6b !important;
            transform: translateY(-2px) !important;
        }
        div[data-testid="stPopover"] > div > button *,
        div[data-testid="stPopover"] button[data-testid="stPopoverButton"] * {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        .rb-header {
            background: #1f2a44;
            color: #ffffff;
            padding: 10px 14px;
            border-radius: 10px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rb-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 13.5px;
        }
        .rb-sub {
            font-size: 10.5px;
            color: #cfd6e6;
        }
        .rb-msg-bot {
            background: #f0ebe2;
            color: #1f2a44;
            padding: 8px 11px;
            border-radius: 10px 10px 10px 3px;
            font-size: 12px;
            line-height: 1.45;
            margin: 5px 0;
        }
        .rb-msg-user {
            background: #1f2a44;
            color: #ffffff;
            padding: 7px 11px;
            border-radius: 10px 10px 3px 10px;
            font-size: 12px;
            line-height: 1.4;
            margin: 5px 0 5px auto;
            max-width: 88%;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.popover("🔖 Ayraç AI · Veriye sor"):
        st.markdown(
            "<div class='rb-header'>"
            "<div><div class='rb-title'>🔖 Ayraç AI · Veriye sor</div>"
            "<div class='rb-sub'>Kitap &amp; Kırtasiye B2B Akıllı Karar Asistanı</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Hızlı soru çipleri
        oneriler = HIZLI_SORULAR.get(sayfa, HIZLI_SORULAR["satis"])
        st.caption("Hızlı sorular (tıkla ve sor):")
        c_rows = [oneriler[:3], oneriler[3:]]
        for r_idx, row_chips in enumerate(c_rows):
            if not row_chips:
                continue
            cols = st.columns(len(row_chips))
            for c_idx, soru_metni in enumerate(row_chips):
                if cols[c_idx].button(soru_metni, key=f"rb_chip_{sayfa}_{r_idx}_{c_idx}", use_container_width=True):
                    yanit = _cevap_uret(soru_metni, bilgi, sayfa)
                    ss["rafbot_msgs"].append({"role": "user", "text": soru_metni})
                    ss["rafbot_msgs"].append({"role": "assistant", "text": yanit})
                    st.rerun()

        # Mesaj geçmişi
        gecmis_kutu = st.container(height=250, border=True)
        with gecmis_kutu:
            for m_item in ss["rafbot_msgs"][-14:]:
                if m_item["role"] == "user":
                    st.markdown(f"<div class='rb-msg-user'>{m_item['text']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='rb-msg-bot'>{m_item['text']}</div>", unsafe_allow_html=True)

        # Soru sorma formu
        with st.form(key=f"rb_form_{sayfa}", clear_on_submit=True, border=False):
            f1, f2 = st.columns([3.2, 1])
            girilen = f1.text_input(
                "Sorunuz",
                placeholder="Örn: Ciro neden düştü, S00172, İstanbul, Polisiye…",
                label_visibility="collapsed",
            )
            gonder = f2.form_submit_button("Sor", type="primary", use_container_width=True)
            if gonder and girilen.strip():
                yanit = _cevap_uret(girilen.strip(), bilgi, sayfa)
                ss["rafbot_msgs"].append({"role": "user", "text": girilen.strip()})
                ss["rafbot_msgs"].append({"role": "assistant", "text": yanit})
                st.rerun()

        alt1, alt2 = st.columns([2, 1])
        alt1.caption("İpucu: Müşteri kodu, il veya kitap adı da yazabilirsin.")
        if alt2.button("Temizle", key=f"rb_temizle_{sayfa}", use_container_width=True):
            ss["rafbot_msgs"] = [ss["rafbot_msgs"][0]]
            st.rerun()
