"""RafRadar Yönetici Özeti, Pazarlama, Tedarik Zinciri ve Veri Kalitesi analitik fonksiyonları."""
from __future__ import annotations

import numpy as np
import pandas as pd
from common import analytics as an


def _m_avg4(wk: np.ndarray, h: int = 8) -> list[float]:
    v = float(np.mean(wk[-4:])) if len(wk) else 0.0
    return [max(0.0, v)] * h


def _m_avg8(wk: np.ndarray, h: int = 8) -> list[float]:
    v = float(np.mean(wk[-8:])) if len(wk) else 0.0
    return [max(0.0, v)] * h


def _m_holt(wk: np.ndarray, h: int = 8) -> list[float]:
    if len(wk) == 0:
        return [0.0] * h
    if len(wk) == 1:
        return [max(0.0, float(wk[0]))] * h
    alpha, beta, phi = 0.35, 0.15, 0.85
    lvl = float(wk[0])
    trd = float(wk[1] - wk[0])
    for y in wk[1:]:
        prev_lvl = lvl
        lvl = alpha * float(y) + (1.0 - alpha) * (lvl + phi * trd)
        trd = beta * (lvl - prev_lvl) + (1.0 - beta) * phi * trd
    out = []
    phi_sum = 0.0
    for k in range(1, h + 1):
        phi_sum += phi ** k
        out.append(max(0.0, lvl + phi_sum * trd))
    return out


MODELLER = [
    ("Son 4 hafta ortalaması", _m_avg4),
    ("Son 8 hafta ortalaması", _m_avg8),
    ("Trend (Holt)", _m_holt),
]

POLITIKA = {
    "AX": "Düzenli otomatik sipariş",
    "AY": "Tahmin + güvenlik stoğu",
    "AZ": "Sık kontrol + yüksek güvenlik stoğu",
    "BX": "Sabit aralıkla sipariş",
    "BY": "Tahmin + orta güvenlik stoğu",
    "BZ": "Siparişe göre tedarik",
    "CX": "Toplu al, az uğraş",
    "CY": "Az stok tut",
    "CZ": "Stok tutma / katalogdan çıkar",
}

SEG_AD = [
    "Şampiyonlar",
    "Sadık",
    "Yeni / Potansiyel",
    "Dikkat gerektiren",
    "Risk altında",
    "Uykuda / Kayıp",
    "Dönemde sipariş vermeyen",
]

SEG_ACIKLAMA = [
    "Yakın zamanda, sık ve yüksek tutarlı sipariş veren en değerli müşteriler.",
    "Düzenli sipariş veren ve ciro katkısı yüksek müşteriler.",
    "Yakın zamanda sipariş vermiş ancak sipariş sıklığı/tutarı henüz düşük müşteriler.",
    "Orta sıklık ve tutarda sipariş veren, hatırlatma ile sadık gruba taşınabilecek müşteriler.",
    "Geçmişte sık ve yüksek tutarlı alım yapıp son dönemde sipariş vermeyi geciktiren kritik müşteriler.",
    "Uzun süredir sipariş vermeyen ve alım sıklığı düşük müşteriler.",
    "Seçili tarih aralığında hiç sipariş vermemiş ancak daha önce alımı bulunan müşteriler.",
]

SEG_RENK = ["#14513f", "#3d9a84", "#8cc5b5", "#e0a526", "#d9622b", "#d0574f", "#cfc9bd"]
SEG_YAZI = ["#fff", "#fff", "#1f2a44", "#1f2a44", "#fff", "#fff", "#1f2a44"]


def seg_of(R: int, FM: int) -> int:
    """R (1..5, 5=en yeni) ve FM (1..5, 5=en yüksek) puanını 0..5 RFM segment indeksine çevirir."""
    if FM >= 4 and R >= 4 and (FM == 5 or R == 5):
        return 0  # Şampiyonlar
    if FM >= 3 and R >= 4:
        return 1  # Sadık
    if FM >= 4 and R == 3:
        return 1  # Sadık
    if FM <= 2 and R >= 4:
        return 2  # Yeni / Potansiyel
    if R == 3 and FM <= 3:
        return 3  # Dikkat gerektiren
    if FM >= 3 and R <= 2:
        return 4  # Risk altında
    return 5  # Uykuda / Kayıp


def baz_tahmin30(m: pd.DataFrame) -> tuple[float, float]:
    """Önümüzdeki 30 gün için baz ciro tahmini ve geriye dönük test WAPE hatasını döndürür."""
    if m.empty:
        return 0.0, 0.25
    min_g, max_g = m["gun"].min(), m["gun"].max()
    gunluk = m.groupby("gun")["net"].sum().reindex(pd.date_range(min_g, max_g), fill_value=0.0).values
    n_haf = len(gunluk) // 7
    if n_haf < 8:
        ort_gun = float(np.mean(gunluk)) if len(gunluk) else 0.0
        return ort_gun * 30.0, 0.25

    wk = gunluk[: n_haf * 7].reshape(-1, 7).sum(axis=1)
    sonuc, en = geri_test(wk)
    best = sonuc[en]
    f5 = best["f"](wk, 5)
    baz = float(sum(f5[:4]) + f5[4] * (2.0 / 7.0))
    wape = float(np.clip(best["wape"], 0.08, 0.45))
    return baz, wape


def norm_asimi(tum: pd.DataFrame, m: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Kategori medyan indirimini 10+ puan aşan siparişleri ve kategori kırılımını hesaplar."""
    hedef = m if m is not None else tum
    if hedef.empty:
        bos = pd.DataFrame(columns=["md", "asan", "pay", "fazla"])
        return {"asan_siparis": 0, "toplam": 1, "fazla": 0.0, "tavan": 0.0}, bos

    baz_oc = tum.dropna(subset=["Category"]).groupby(["InvoiceNo", "Category"], as_index=False).agg(
        g=("gross", "sum"),
        n=("net", "sum"),
    )
    baz_oc["disc"] = np.where(baz_oc["g"] > 0, 1.0 - baz_oc["n"] / baz_oc["g"], 0.0)
    kat_md = baz_oc.groupby("Category")["disc"].median()

    oc = hedef.dropna(subset=["Category"]).groupby(["InvoiceNo", "Category"], as_index=False).agg(
        g=("gross", "sum"),
        n=("net", "sum"),
    )
    oc["disc"] = np.where(oc["g"] > 0, 1.0 - oc["n"] / oc["g"], 0.0)
    oc["md"] = oc["Category"].map(kat_md).fillna(0.38)
    oc["sapma"] = oc["disc"] - oc["md"]
    oc["asan"] = oc["sapma"] > 0.10
    oc["fazla"] = np.where(oc["asan"], oc["sapma"] * oc["g"], 0.0)
    oc["tavan"] = np.where(oc["sapma"] > 0.05, (oc["sapma"] - 0.05) * oc["g"], 0.0)

    kat_ozet = oc.groupby("Category").agg(
        md=("md", "first"),
        asan=("asan", "sum"),
        toplam=("InvoiceNo", "count"),
        fazla=("fazla", "sum"),
    )
    kat_ozet["pay"] = kat_ozet["asan"] / np.maximum(1, kat_ozet["toplam"])
    kat_ozet = kat_ozet.sort_values("pay", ascending=False)

    sip_bazli = oc.groupby("InvoiceNo").agg(asan=("asan", "any"), fazla=("fazla", "sum"), tavan=("tavan", "sum"))
    top = {
        "asan_siparis": int(sip_bazli["asan"].sum()),
        "toplam": max(1, int(len(sip_bazli))),
        "fazla": float(oc["fazla"].sum()),
        "tavan": float(oc["tavan"].sum()),
    }
    return top, kat_ozet


def abc_xyz(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ürünleri satılan adede göre ABC ve aylık talep oynaklığına (CV) göre XYZ olarak sınıflandırır."""
    if df.empty:
        bos = pd.Series(dtype=float)
        return bos, bos, pd.Series(dtype=str), pd.Series(dtype=str)

    qadet = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False)
    toplam = max(1e-9, float(qadet.sum()))
    kum = qadet.cumsum() / toplam
    onceki = kum.shift(1, fill_value=0.0)
    abc = pd.Series(np.where(onceki < 0.80, "A", np.where(onceki < 0.95, "B", "C")), index=qadet.index)

    ay_mat = (
        df.assign(ay=df["gun"].dt.month)
        .pivot_table(index="Description", columns="ay", values="Quantity", aggfunc="sum", fill_value=0.0)
        .reindex(index=qadet.index, columns=range(1, 13), fill_value=0.0)
    )
    ort = ay_mat.mean(axis=1).replace(0, np.nan)
    std = ay_mat.std(axis=1, ddof=0)
    cv = (std / ort).fillna(2.0)
    xyz = pd.Series(np.where(cv <= 0.50, "X", np.where(cv <= 1.00, "Y", "Z")), index=qadet.index)
    return qadet, kum, abc, xyz


def yavas_urunler(m: pd.DataFrame, bitis: str) -> pd.DataFrame:
    """Bitiş tarihinden önceki 90 günde satışı bir önceki 90 güne göre %70 ve üzeri düşen ürünleri döndürür."""
    if m.empty:
        return pd.DataFrame(columns=["q90", "pr", "d"])
    tb = pd.Timestamp(bitis)
    t1 = tb - pd.Timedelta(days=89)
    t0 = t1 - pd.Timedelta(days=90)

    son90 = m[(m["gun"] >= t1) & (m["gun"] <= tb)].groupby("Description")["Quantity"].sum()
    onc90 = m[(m["gun"] >= t0) & (m["gun"] < t1)].groupby("Description")["Quantity"].sum()
    u_ozet = m.groupby("Description").agg(gross=("gross", "sum"), net=("net", "sum"), q=("Quantity", "sum"))

    df_y = pd.DataFrame({"q_onc": onc90, "q_son": son90}).fillna(0.0)
    df_y = df_y[(df_y["q_onc"] >= 20) & (df_y["q_son"] <= df_y["q_onc"] * 0.30)].copy()
    if df_y.empty:
        return pd.DataFrame(columns=["q90", "pr", "d"])

    u_sub = u_ozet.reindex(df_y.index)
    df_y["q90"] = np.maximum(df_y["q_son"], df_y["q_onc"] * 0.15)
    df_y["pr"] = (u_sub["gross"] / np.maximum(1.0, u_sub["q"])).fillna(35.0)
    df_y["d"] = np.clip(1.0 - u_sub["net"] / np.maximum(1.0, u_sub["gross"]), 0.05, 0.75).fillna(0.38)
    return df_y[["q90", "pr", "d"]]


def musteri_degisim(tum: pd.DataFrame, m: pd.DataFrame, a, b, ca, cb) -> dict:
    """Seçili dönem ve karşılaştırma dönemi arasındaki aktif, yeni, geri dönen ve kaybedilen müşterileri hesaplar."""
    s_df = an.araliga_kes(m, a, b)
    c_df = an.araliga_kes(m, ca, cb)
    ilk_tum = tum.groupby("CustomerID")["gun"].min()

    set_s = set(s_df["CustomerID"].unique())
    set_c = set(c_df["CustomerID"].unique())

    ct = len(set_s & set_c)
    ls = len(set_c - set_s)
    gelenler = set_s - set_c

    ta = pd.Timestamp(a)
    tca = pd.Timestamp(ca)
    nw = sum(1 for cid in gelenler if ilk_tum.get(cid, ta) >= ta)
    bk = len(gelenler) - nw
    nwC = sum(1 for cid in set_c if ilk_tum.get(cid, tca) >= tca)

    return {
        "aS": len(set_s),
        "aC": len(set_c),
        "nw": nw,
        "nwC": max(1, nwC) if len(set_c) and nwC == 0 else nwC,
        "bk": bk,
        "ls": ls,
        "ct": ct,
    }


def aylik(m: pd.DataFrame, yil: int) -> pd.DataFrame | None:
    d = m[m["gun"].dt.year == int(yil)]
    if d.empty:
        return None
    g = d.groupby(d["gun"].dt.month).agg(net=("net", "sum"), gross=("gross", "sum")).reindex(range(1, 13), fill_value=0.0)
    g["disc"] = np.where(g["gross"] > 0, 1.0 - g["net"] / g["gross"], 0.0)
    return g


def kohort_ceyrek(m: pd.DataFrame) -> tuple[list, dict]:
    """Çeyreklik müşteri kohort elde tutma (retention) tablosunu ve ortalamalarını hesaplar."""
    if m.empty:
        return [], {1: None, 2: None, 3: None}

    yil = int(m["gun"].dt.year.mode().iloc[0])
    d = m[m["gun"].dt.year == yil]
    if d.empty:
        return [], {1: None, 2: None, 3: None}

    d_q = d.assign(q=d["gun"].dt.quarter)
    mevcut_q = sorted(d_q["q"].unique().tolist())
    max_q = max(mevcut_q)
    ilk_q = d_q.groupby("CustomerID")["q"].min()
    aktif_q = d_q.groupby(["CustomerID", "q"]).size().unstack(fill_value=0) > 0

    ks = []
    top_pay = {1: 0, 2: 0, 3: 0}
    top_payda = {1: 0, 2: 0, 3: 0}

    for q0 in range(1, 5):
        cids = ilk_q[ilk_q == q0].index
        n = len(cids)
        if n == 0:
            continue
        yuz = []
        for k in (1, 2, 3):
            qk = q0 + k
            if qk <= max_q and qk in aktif_q.columns:
                tutulan = int(aktif_q.loc[cids, qk].sum())
                yuz.append(float(tutulan / n * 100.0))
                top_pay[k] += tutulan
                top_payda[k] += n
            else:
                yuz.append(None)
        ks.append((f"{yil} Ç{q0}", n, yuz))

    kg = {k: (float(top_pay[k] / top_payda[k] * 100.0) if top_payda[k] > 0 else None) for k in (1, 2, 3)}
    return ks, kg


def rfm(tum: pd.DataFrame, m: pd.DataFrame, a, b) -> tuple[pd.DataFrame, pd.Index]:
    """Seçili dönemde sipariş veren müşteriler için RFM puanlarını ve segmentlerini hesaplar."""
    mr = an.araliga_kes(m, a, b)
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    onceden_all = m[m["gun"] < ta]["CustomerID"].unique()

    if mr.empty:
        bos = pd.DataFrame(columns=["son", "sip", "net", "rec", "R", "FM", "seg"])
        return bos, pd.Index(onceden_all)

    g = mr.groupby("CustomerID").agg(
        son=("gun", "max"),
        sip=("InvoiceNo", "nunique"),
        net=("net", "sum"),
    )
    g["rec"] = (tb - g["son"]).dt.days.astype(int)

    n = len(g)
    if n >= 5:
        r_score = pd.qcut(g["rec"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        f_score = pd.qcut(g["sip"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        m_score = pd.qcut(g["net"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    else:
        r_score = pd.Series(3, index=g.index)
        f_score = pd.Series(3, index=g.index)
        m_score = pd.Series(3, index=g.index)

    fm_score = np.clip(np.round((f_score + m_score) / 2.0), 1, 5).astype(int)
    g["R"] = r_score
    g["FM"] = fm_score
    g["seg"] = [seg_of(int(r), int(fm)) for r, fm in zip(g["R"], g["FM"])]

    onceden = pd.Index([c for c in onceden_all if c not in g.index])
    return g, onceden


def gecikme(tum: pd.DataFrame, rf: pd.DataFrame, b) -> pd.Series:
    """Her müşterinin kendi tipik sipariş aralığına göre gecikme durumunu (0..3) hesaplar."""
    tb = pd.Timestamp(b)
    gecmis = tum[tum["gun"] <= tb]
    if gecmis.empty:
        return pd.Series(dtype=int)

    gun_list = gecmis[["CustomerID", "gun"]].drop_duplicates().sort_values(["CustomerID", "gun"])
    gun_list["fark"] = gun_list.groupby("CustomerID")["gun"].diff().dt.days

    ozet = gun_list.groupby("CustomerID").agg(
        gun_say=("gun", "count"),
        son=("gun", "max"),
        tipik=("fark", "median"),
    )
    ozet["rec"] = (tb - ozet["son"]).dt.days
    oran = ozet["rec"] / np.maximum(1.0, ozet["tipik"].fillna(999.0))

    durum = np.where(
        ozet["gun_say"] < 3,
        3,
        np.where(oran <= 1.0, 0, np.where(oran <= 2.0, 1, 2)),
    )
    return pd.Series(durum, index=ozet.index, dtype=int)


def il_dagilimi(tum: pd.DataFrame, mr: pd.DataFrame) -> pd.DataFrame:
    if mr.empty:
        return pd.DataFrame(columns=["net", "musteri", "siparis"])
    g = mr.groupby("City").agg(
        net=("net", "sum"),
        musteri=("CustomerID", "nunique"),
        siparis=("InvoiceNo", "nunique"),
    )
    return g.sort_values("net", ascending=False)


def kategori_endeksi(mr: pd.DataFrame, rf: pd.DataFrame) -> tuple[list[str], list[list[int]]]:
    """RFM segmentlerinin en büyük 6 kategoriyi alım eğilim endeksini (100 = ortalama) hesaplar."""
    if mr.empty or rf.empty:
        return [], []
    df_k = mr.dropna(subset=["Category"]).copy()
    if df_k.empty:
        return [], []

    cats = df_k.groupby("Category")["net"].sum().nlargest(6).index.tolist()
    df_k["seg"] = df_k["CustomerID"].map(rf["seg"])
    df_k = df_k.dropna(subset=["seg"])
    if df_k.empty:
        return [], []

    toplam_net = max(1e-9, float(df_k["net"].sum()))
    genel_pay = df_k.groupby("Category")["net"].sum() / toplam_net

    mat = []
    for s in range(6):
        sub = df_k[df_k["seg"] == s]
        s_top = float(sub["net"].sum())
        if s_top <= 0:
            mat.append([100 for _ in cats])
            continue
        s_pay = sub.groupby("Category")["net"].sum() / s_top
        satir = [int(round(float(s_pay.get(c, 0.0) / max(1e-9, genel_pay.get(c, 1e-9)) * 100.0))) for c in cats]
        mat.append(satir)
    return cats, mat


def haftalik(tum: pd.DataFrame, urun: str) -> tuple[np.ndarray, np.ndarray]:
    min_g, max_g = tum["gun"].min(), tum["gun"].max()
    idx = pd.date_range(min_g, max_g, freq="D")
    u_df = tum[tum["Description"] == urun]
    gunluk = u_df.groupby("gun")["Quantity"].sum().reindex(idx, fill_value=0.0).values.astype(float)
    n_haf = len(gunluk) // 7
    wk = gunluk[: n_haf * 7].reshape(-1, 7).sum(axis=1)
    return wk, gunluk


def geri_test(wk: np.ndarray) -> tuple[list[dict], int]:
    """Son 12 haftada 3 tahmin modelini geriye dönük test eder ve en düşük WAPE'li modeli seçer."""
    n = len(wk)
    test_len = min(12, max(4, n // 3))
    bas = n - test_len

    sonuc = []
    for ad, fn in MODELLER:
        fcs = []
        err4 = 0.0
        act4 = 0.0
        for t in range(bas, n):
            gecmis = wk[:t]
            tah_1 = fn(gecmis, 1)[0]
            fcs.append(float(tah_1))
            if t + 4 <= n:
                tah_4 = sum(fn(gecmis, 4))
                ger_4 = float(np.sum(wk[t : t + 4]))
                err4 += abs(tah_4 - ger_4)
                act4 += ger_4
        fcs_arr = np.array(fcs, dtype=float)
        ger_arr = wk[bas:n]
        if act4 > 0:
            wape = float(err4 / act4)
        else:
            wape = float(np.sum(np.abs(fcs_arr - ger_arr)) / max(1.0, float(np.sum(ger_arr))))
        sd = float(np.std(ger_arr - fcs_arr, ddof=0)) if len(ger_arr) else 1.0
        sonuc.append({"ad": ad, "f": fn, "wape": wape, "sd": max(1.0, sd), "fcs": fcs_arr.tolist()})

    en = int(np.argmin([r["wape"] for r in sonuc]))
    return sonuc, en


def _z_degeri(sl: int) -> float:
    tablo = {
        85: 1.036, 86: 1.080, 87: 1.126, 88: 1.175, 89: 1.227,
        90: 1.282, 91: 1.341, 92: 1.405, 93: 1.476, 94: 1.555,
        95: 1.645, 96: 1.751, 97: 1.881, 98: 2.054, 99: 2.326,
    }
    return tablo.get(int(sl), 1.645)


def stok(wk: np.ndarray, gunluk: np.ndarray, lt: int, sl: int) -> dict:
    """Güvenlik stoğu, sipariş noktası ve son 26 haftalık günlük stok simülasyonunu hesaplar."""
    son_wk = wk[-26:] if len(wk) >= 26 else wk
    mu = float(np.mean(son_wk)) if len(son_wk) else 0.0
    sd = float(np.std(son_wk, ddof=0)) if len(son_wk) else 0.0
    z = _z_degeri(sl)

    ss = float(z * sd * np.sqrt(lt))
    rop = float(mu * lt + ss)

    sim_gun = gunluk[-182:] if len(gunluk) >= 182 else gunluk
    n_gun = len(sim_gun)
    sip_miktari = max(1.0, mu * 4.0)
    lt_gun = max(1, int(lt) * 7)

    eldeki = rop + mu * 2.0
    yoldaki: list[tuple[int, float]] = []
    ser = []
    outd = []

    for g_idx, talep in enumerate(sim_gun):
        # Teslimatları ekle
        yeni_yoldaki = []
        for varis_gunu, mik in yoldaki:
            if varis_gunu <= g_idx:
                eldeki += mik
            else:
                yeni_yoldaki.append((varis_gunu, mik))
        yoldaki = yeni_yoldaki

        # Günlük satış
        if eldeki >= talep:
            eldeki -= talep
        else:
            eldeki = 0.0
            outd.append(g_idx)

        ser.append(float(eldeki))

        # Sipariş noktası kontrolü
        toplam_pozisyon = eldeki + sum(m for _, m in yoldaki)
        if toplam_pozisyon <= rop:
            yoldaki.append((g_idx + lt_gun, sip_miktari))

    return {
        "ss": ss,
        "rop": rop,
        "mu": mu,
        "out": len(outd),
        "n": n_gun,
        "outd": outd,
        "ser": ser,
    }


def mevsimsellik(tum: pd.DataFrame) -> tuple[list[str], list[list[int]], int]:
    """En çok satan 8 kategorinin 12 aylık mevsimsellik endeksini (100 = ortalama) döndürür."""
    if tum.empty:
        return [], [], 2020
    yil = int(tum["gun"].dt.year.mode().iloc[0])
    d = tum[(tum["gun"].dt.year == yil) & tum["Category"].notna()]
    if d.empty:
        return [], [], yil

    cats = d.groupby("Category")["Quantity"].sum().nlargest(8).index.tolist()
    piv = (
        d[d["Category"].isin(cats)]
        .assign(ay=d["gun"].dt.month)
        .pivot_table(index="Category", columns="ay", values="Quantity", aggfunc="sum", fill_value=0.0)
        .reindex(index=cats, columns=range(1, 13), fill_value=0.0)
    )
    ort = piv.mean(axis=1).replace(0, 1.0)
    endeks = piv.div(ort, axis=0) * 100.0
    sm = [[int(round(v)) for v in endeks.loc[c].values] for c in cats]
    return cats, sm, yil


def satissiz(tum: pd.DataFrame) -> pd.DataFrame:
    """Her ürünün veri bitiş tarihi itibarıyla kaç gündür satılmadığını hesaplar."""
    if tum.empty:
        return pd.DataFrame(columns=["gun", "kategori", "adet", "net"])
    bit = tum["gun"].max()
    g = tum.groupby("Description").agg(
        son=("gun", "max"),
        kategori=("Category", "first"),
        adet=("Quantity", "sum"),
        net=("net", "sum"),
    )
    g["kategori"] = g["kategori"].fillna("Belirsiz")
    g["gun"] = (bit - g["son"]).dt.days.astype(int)
    return g[["gun", "kategori", "adet", "net"]]
