"""RafRadar Satış & Fiyat ve temel analitik hesaplama fonksiyonları."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def araliga_kes(df: pd.DataFrame, a, b) -> pd.DataFrame:
    if df.empty:
        return df
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    return df[(df["gun"] >= ta) & (df["gun"] <= tb)]


def ozet(m: pd.DataFrame, a, b) -> dict:
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    gun = max(1, int((tb - ta).days) + 1)
    x = araliga_kes(m, ta, tb)
    if x.empty:
        return {
            "gun": gun, "net": 0.0, "gross": 0.0, "q": 0.0, "ord": 0, "musteri": 0,
            "disc": 0.0, "dnet": 0.0, "dq": 0.0, "dord": 0.0, "dcust": 0.0,
            "b": 0.0, "basket": 0.0, "L": 0.0, "dd": 0.0, "dbq": 0.0, "dbt": 0.0,
        }

    net = float(x["net"].sum())
    gross = float(x["gross"].sum())
    q = float(x["Quantity"].sum())
    ord_cnt = int(x["InvoiceNo"].nunique())
    musteri = int(x["CustomerID"].nunique())
    disc = float(1.0 - net / gross) if gross > 0 else 0.0

    dnet = net / gun
    dq = q / gun
    dord = ord_cnt / gun

    g = x.groupby("gun").agg(
        net=("net", "sum"),
        gross=("gross", "sum"),
        q=("Quantity", "sum"),
        o=("InvoiceNo", "nunique"),
        c=("CustomerID", "nunique"),
    )
    dcust = float(g["c"].sum() / gun)
    dd = float((1.0 - g["net"] / g["gross"].replace(0, np.nan)).dropna().mean()) if len(g) else disc
    dbq = float((g["q"] / g["o"].replace(0, np.nan)).dropna().mean()) if len(g) else 0.0
    dbt = float((g["net"] / g["o"].replace(0, np.nan)).dropna().mean()) if len(g) else 0.0

    b_val = q / ord_cnt if ord_cnt > 0 else 0.0
    basket = net / ord_cnt if ord_cnt > 0 else 0.0
    L = gross / q if q > 0 else 0.0

    return {
        "gun": gun,
        "net": net,
        "gross": gross,
        "q": q,
        "ord": ord_cnt,
        "musteri": musteri,
        "disc": disc,
        "dnet": dnet,
        "dq": dq,
        "dord": dord,
        "dcust": dcust,
        "b": b_val,
        "basket": basket,
        "L": L,
        "dd": dd,
        "dbq": dbq,
        "dbt": dbt,
    }


def kopru(S: dict, C: dict) -> list[float]:
    """Günlük ciro farkını (S['dnet'] - C['dnet']) 4 çarpımsal faktöre (LMDI) ayrıştırır."""
    d1, d0 = float(S["dnet"]), float(C["dnet"])
    if d1 <= 0 or d0 <= 0:
        return [d1 - d0, 0.0, 0.0, 0.0]
    if abs(d1 - d0) < 1e-6:
        return [0.0, 0.0, 0.0, 0.0]

    f0 = [max(1e-9, C["dord"]), max(1e-9, C["b"]), max(1e-9, C["L"]), max(1e-9, 1.0 - C["disc"])]
    f1 = [max(1e-9, S["dord"]), max(1e-9, S["b"]), max(1e-9, S["L"]), max(1e-9, 1.0 - S["disc"])]
    l_mean = (d1 - d0) / (np.log(d1) - np.log(d0))
    return [float(l_mean * np.log(a / b)) for a, b in zip(f1, f0)]


def urun_listesi(tum: pd.DataFrame, n: int = 50) -> list[str]:
    if tum.empty:
        return []
    return tum.groupby("Description")["net"].sum().nlargest(n).index.tolist()


def urun_referans(tum: pd.DataFrame, urun: str, s_gun: int, ca, cb) -> dict:
    u_df = tum[tum["Description"] == urun]
    if u_df.empty:
        return {"q": 0.0, "n": 0.0, "g": 0.0, "cap": 0.45, "mx": 0.50, "fb": True}

    cap = float(np.percentile(u_df["disc"], 95))
    mx = float(u_df["disc"].max())

    u_comp = araliga_kes(u_df, ca, cb)
    c_gun = max(1, int((pd.Timestamp(cb) - pd.Timestamp(ca)).days) + 1)
    if not u_comp.empty and u_comp["Quantity"].sum() > 0:
        oran = s_gun / c_gun
        return {
            "q": float(u_comp["Quantity"].sum() * oran),
            "n": float(u_comp["net"].sum() * oran),
            "g": float(u_comp["gross"].sum() * oran),
            "cap": cap,
            "mx": mx,
            "fb": False,
        }

    tot_gun = max(1, int((tum["gun"].max() - tum["gun"].min()).days) + 1)
    oran = s_gun / tot_gun
    return {
        "q": float(u_df["Quantity"].sum() * oran),
        "n": float(u_df["net"].sum() * oran),
        "g": float(u_df["gross"].sum() * oran),
        "cap": cap,
        "mx": mx,
        "fb": True,
    }


def hafta_kategori(mr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if mr.empty:
        return {"Ciro payı": pd.DataFrame(), "Sipariş adedi": pd.DataFrame()}
    top6 = mr.dropna(subset=["Category"]).groupby("Category")["net"].sum().nlargest(6).index.tolist()
    sub = mr[mr["Category"].isin(top6)]
    piv_net = (
        sub.pivot_table(index="Category", columns="dow", values="net", aggfunc="sum", fill_value=0.0)
        .reindex(index=top6, columns=range(7), fill_value=0.0)
    )
    row_sums = piv_net.sum(axis=1).replace(0, 1.0)
    ciro_payi = piv_net.div(row_sums, axis=0) * 100.0

    piv_sip = (
        sub.pivot_table(index="Category", columns="dow", values="InvoiceNo", aggfunc="nunique", fill_value=0)
        .reindex(index=top6, columns=range(7), fill_value=0)
    )
    return {"Ciro payı": ciro_payi, "Sipariş adedi": piv_sip}


def birlikte(df: pd.DataFrame, limit: int = 3) -> list[dict]:
    """Aynı siparişte en sık birlikte alınan ürün çiftlerini döndürür."""
    if df.empty:
        return []
    top_u = df.groupby("Description")["InvoiceNo"].nunique().nlargest(100)
    if len(top_u) < 2:
        return []
    urunler = top_u.index.tolist()
    sub = df[df["Description"].isin(urunler)][["InvoiceNo", "Description"]].drop_duplicates()

    # İkili matris çarpımı ile hızlı ortak sipariş sayımı
    cat_u = pd.Categorical(sub["Description"], categories=urunler, ordered=True)
    cat_inv = pd.Categorical(sub["InvoiceNo"])
    n_inv = len(cat_inv.categories)
    n_u = len(urunler)
    if n_inv == 0 or n_u < 2:
        return []

    mat = np.zeros((n_inv, n_u), dtype=np.int16)
    mat[cat_inv.codes, cat_u.codes] = 1
    co = mat.T.astype(np.int32) @ mat.astype(np.int32)
    diag = np.diag(co).copy()
    np.fill_diagonal(co, 0)

    ciftler = []
    for i in range(n_u):
        for j in range(i + 1, n_u):
            n_co = int(co[i, j])
            if n_co >= 3:
                payda = max(1, min(int(diag[i]), int(diag[j])))
                ciftler.append({
                    "a": urunler[i],
                    "b": urunler[j],
                    "n": n_co,
                    "oran": float(n_co / payda),
                })
    ciftler.sort(key=lambda x: (-x["n"], -x["oran"]))
    return ciftler[:limit]


def en_cok_satanlar(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    top_q = df.groupby("Description")["Quantity"].sum().nlargest(n)
    satirlar = []
    for sira, (u, adet) in enumerate(top_q.items(), 1):
        invs = set(df.loc[df["Description"] == u, "InvoiceNo"].unique())
        n_inv = max(1, len(invs))
        diger = df[(df["InvoiceNo"].isin(invs)) & (df["Description"] != u)]
        if not diger.empty:
            es_top = diger.groupby("Description")["InvoiceNo"].nunique().nlargest(1)
            es_urun = str(es_top.index[0])
            oran = float(es_top.iloc[0] / n_inv)
        else:
            es_urun = "—"
            oran = 0.0
        satirlar.append({
            "#": sira,
            "Best seller": u,
            "Adet": float(adet),
            "Birlikte en çok alınan": es_urun,
            "Birlikte alma oranı": oran,
        })
    return pd.DataFrame(satirlar)


def pareto(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["net", "kum", "seg"])
    s = df.groupby(col)["net"].sum().sort_values(ascending=False)
    toplam = max(1e-9, float(s.sum()))
    kum = s.cumsum() / toplam
    onceki = kum.shift(1, fill_value=0.0)
    seg = np.where(onceki < 0.80, "A", np.where(onceki < 0.95, "B", "C"))
    return pd.DataFrame({"net": s, "kum": kum, "seg": seg}, index=s.index)


def musteri_ozeti(tum: pd.DataFrame, mr: pd.DataFrame, pc: pd.DataFrame, max_gun: pd.Timestamp) -> pd.DataFrame:
    if pc.empty:
        return pd.DataFrame(columns=["seg", "net", "kum", "il", "siparis", "son", "pasif"])
    out = pc.copy()
    g_mr = mr.groupby("CustomerID").agg(siparis=("InvoiceNo", "nunique"))
    son_tum = tum.groupby("CustomerID")["gun"].max()
    il_tum = tum.groupby("CustomerID")["City"].first()

    out["il"] = il_tum.reindex(out.index).fillna("")
    out["siparis"] = g_mr["siparis"].reindex(out.index).fillna(0).astype(int)
    out["son"] = son_tum.reindex(out.index)
    out["pasif"] = (pd.Timestamp(max_gun) - out["son"]).dt.days
    return out


def _musteri_ozellikleri(df_gecmis: pd.DataFrame, ref_gun: pd.Timestamp) -> pd.DataFrame:
    g = df_gecmis.groupby("CustomerID").agg(
        ilk=("gun", "min"),
        son=("gun", "max"),
        sip=("InvoiceNo", "nunique"),
        gun_say=("gun", "nunique"),
        net=("net", "sum"),
        adet=("Quantity", "sum"),
        kat=("Category", "nunique"),
    )
    g["rec"] = (ref_gun - g["son"]).dt.days.astype(float)
    g["span"] = np.maximum(1.0, (g["son"] - g["ilk"]).dt.days.astype(float))
    g["aralik"] = np.where(g["gun_say"] > 1, g["span"] / (g["gun_say"] - 1), 90.0)
    g["gecikme_orani"] = g["rec"] / np.maximum(7.0, g["aralik"])
    g["sepet"] = g["net"] / np.maximum(1.0, g["sip"])

    son30 = df_gecmis[df_gecmis["gun"] > (ref_gun - pd.Timedelta(days=30))].groupby("CustomerID")["net"].sum()
    son90 = df_gecmis[df_gecmis["gun"] > (ref_gun - pd.Timedelta(days=90))].groupby("CustomerID")["net"].sum()
    g["net30"] = son30.reindex(g.index).fillna(0.0)
    g["net90"] = son90.reindex(g.index).fillna(0.0)
    return g


def churn_modeli(df: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Gradient Boosting ile 30 günlük sipariş vermeme (churn) olasılığını ve beklenen 30 günlük ciroyu hesaplar."""
    max_gun = df["gun"].max()
    min_gun = df["gun"].min()
    oz_son = _musteri_ozellikleri(df, max_gun)
    cols = ["rec", "sip", "gun_say", "net", "adet", "kat", "aralik", "gecikme_orani", "sepet", "net30", "net90"]

    auc = None
    model = None
    if (max_gun - min_gun).days >= 90:
        t_kes = max_gun - pd.Timedelta(days=30)
        t_val = max_gun - pd.Timedelta(days=60)
        df_egit = df[df["gun"] <= t_val]
        df_hedef_egit = df[(df["gun"] > t_val) & (df["gun"] <= t_kes)]
        df_test = df[df["gun"] <= t_kes]
        df_hedef_test = df[df["gun"] > t_kes]

        if not df_egit.empty and not df_test.empty:
            x_tr = _musteri_ozellikleri(df_egit, t_val)
            aktif_tr = set(df_hedef_egit["CustomerID"].unique())
            y_tr = (~x_tr.index.isin(aktif_tr)).astype(int)

            x_te = _musteri_ozellikleri(df_test, t_kes)
            aktif_te = set(df_hedef_test["CustomerID"].unique())
            y_te = (~x_te.index.isin(aktif_te)).astype(int)

            if len(np.unique(y_tr)) == 2 and len(np.unique(y_te)) == 2:
                clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.07, random_state=42)
                clf.fit(x_tr[cols], y_tr)
                p_te = clf.predict_proba(x_te[cols])[:, 1]
                auc = float(roc_auc_score(y_te, p_te))
                # Tüm etiketli geçmişle (t_kes) yeniden eğit
                clf.fit(x_te[cols], y_te)
                model = clf

    if model is not None:
        p = model.predict_proba(oz_son[cols])[:, 1]
    else:
        # Yedek analitik lojistik skor
        z = (oz_son["rec"] - 35.0) / 25.0 + 0.4 * np.clip(oz_son["gecikme_orani"] - 1.0, -2.0, 4.0) - 0.15 * np.log1p(oz_son["sip"])
        p = 1.0 / (1.0 + np.exp(-z))

    risk = pd.Series(np.where(p >= 0.70, "Yüksek", np.where(p >= 0.40, "Orta", "Düşük")), index=oz_son.index)
    toplam_gun = max(30.0, float((max_gun - min_gun).days + 1))
    m30 = np.maximum(oz_son["net90"] / 3.0, oz_son["net"] * (30.0 / toplam_gun))

    out = pd.DataFrame({"p": p, "risk": risk, "m30": m30}, index=oz_son.index)
    return out, auc


def supheli(tum: pd.DataFrame, mr: pd.DataFrame, tur: str = "indirim") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kategori medyanından sapan şüpheli sipariş-kategori satırlarını döndürür."""
    if mr.empty:
        bos = pd.DataFrame(columns=["InvoiceNo", "Category", "musteri", "q", "g", "n", "indirim", "md", "sapma", "fazla", "mq", "qr"])
        return bos, bos

    baz_df = tum.dropna(subset=["Category"])
    baz_oc = baz_df.groupby(["InvoiceNo", "Category"], as_index=False).agg(
        q=("Quantity", "sum"),
        g=("gross", "sum"),
        n=("net", "sum"),
    )
    baz_oc["indirim"] = np.where(baz_oc["g"] > 0, 1.0 - baz_oc["n"] / baz_oc["g"], 0.0)
    kat_med = baz_oc.groupby("Category").agg(md=("indirim", "median"), mq=("q", "median"))

    oc = mr.dropna(subset=["Category"]).groupby(["InvoiceNo", "Category"], as_index=False).agg(
        musteri=("CustomerID", "first"),
        q=("Quantity", "sum"),
        g=("gross", "sum"),
        n=("net", "sum"),
    )
    oc["indirim"] = np.where(oc["g"] > 0, 1.0 - oc["n"] / oc["g"], 0.0)
    oc = oc.merge(kat_med, on="Category", how="left")
    oc["md"] = oc["md"].fillna(0.38)
    oc["mq"] = oc["mq"].fillna(10.0).replace(0, 1.0)
    oc["sapma"] = oc["indirim"] - oc["md"]
    oc["fazla"] = np.where(oc["sapma"] > 0, oc["sapma"] * oc["g"], 0.0)
    oc["qr"] = oc["q"] / oc["mq"]

    if tur == "indirim":
        sec = oc[oc["sapma"] > 0.10].sort_values("fazla", ascending=False).copy()
    else:
        sec = oc[oc["qr"] >= 5.0].sort_values("qr", ascending=False).copy()
    return sec, oc
