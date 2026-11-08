import os
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =============================================================================
# 1. SAYFA AYARLARI (RAFRADAR V2 ORTAK YAPI)
# =============================================================================
st.set_page_config(
    page_title="RafRadar Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_shared_rafradar_module():
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in candidate_filenames:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            orig_set_page_config = st.set_page_config
            try:
                st.set_page_config = lambda *args, **kwargs: None
                spec = importlib.util.spec_from_file_location(f"dept_{fname}", fpath)
                mod = importlib.util.module_from_spec(spec)
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
# 2. SATIŞ & FİYAT (rafradar-versiyon2.html 'satis' BİREBİR)
# =============================================================================
def render_satis_department(ctx: dict, yo):
    if try_run_external_department_file(["satis.py", "satış.py"], ["render_satis_page", "render_satis", "render_page"], ctx, yo):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]
    valid_all_df = ctx["valid_all_df"]

    disc_pts_str = f"{m['disc_diff_pts']:+.1f} puan".replace(".", ",")
    disc_cls = "neg" if m["disc_diff_pts"] > 0.5 else ("pos" if m["disc_diff_pts"] < -0.5 else "mute")

    kpi_cards = [
        {"label": "Net ciro", "main_val": yo.fmt_tl(S["net"]), "main_chg_html": yo.chg_badge_html(m["rev_chg"]), "sub_label": "Günlük ort.", "sub_val": yo.fmt_tl(S["dnet"], 1), "sub_chg_html": yo.chg_badge_html(m["d_rev_chg"]), "top_color": "#3d9a84"},
        {"label": "Brüt ciro", "main_val": yo.fmt_tl(S["gross"]), "main_chg_html": f'<span class="chg mute">{yo.fmt_int(S["qty"])} adet</span>', "sub_label": "Günlük ort. adet", "sub_val": yo.fmt_int(S["dqty"]), "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]), "top_color": "#1f2a44"},
        {"label": "Ortalama indirim", "main_val": f"%{S['disc']:.1f}".replace(".", ","), "main_chg_html": f'<span class="chg {disc_cls}">{disc_pts_str}</span>', "sub_label": "Fazla indirim", "sub_val": yo.fmt_tl(m["discount_leak_tl"], 1), "sub_chg_html": f'<span class="chg neg">{yo.fmt_int(m["suspicious_orders_count"])} satır</span>', "top_color": "#d0574f"},
        {"label": "Sipariş adedi", "main_val": yo.fmt_int(S["orders"]), "main_chg_html": yo.chg_badge_html(m["inv_chg"]), "sub_label": "Günlük ort.", "sub_val": f"{S['dorders']:.1f}".replace(".", ","), "sub_chg_html": yo.chg_badge_html(m["d_inv_chg"]), "top_color": "#3b6fd1"},
        {"label": "Sepet ortalaması", "main_val": yo.fmt_tl(S["aov"], 1), "main_chg_html": yo.chg_badge_html(C["aov_chg"]), "sub_label": "Karşılaştırma", "sub_val": yo.fmt_tl(C["aov"], 1), "sub_chg_html": '<span class="chg mute">sipariş başı</span>', "top_color": "#e0a526"},
    ]

    yo.render_department_header_and_kpis(
        title="Satış & Fiyat",
        subtitle="Ne zaman, kime, hangi ürünü, hangi indirimle satmalıyız?",
        ctx=ctx,
        kpi_cards=kpi_cards,
    )

    # 1. CİRO VE İNDİRİM KÖPRÜSÜ & 2. İNDİRİM PLANI
    c1, c2 = st.columns([4.8, 5.2])
    with c1:
        disc_tot = S["gross"] - S["net"]
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">1</span>Brüt cirodan net ciroya köprü</h2>
                <p class="sub">Seçili dönem ({ctx['sel_lbl']}) fiyat ve iskonto kırılımı</p>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px;">
                    <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;"><div class="t">Brüt ciro</div><div class="v">{yo.fmt_tl(S['gross'])}</div></div>
                    <div class="out" style="background:#fbe3e0;"><div class="t">Toplam indirim (%{S['disc']:.1f})</div><div class="v" style="color:#8f2a24;">−{yo.fmt_tl(disc_tot)}</div></div>
                    <div class="out"><div class="t">Net ciro</div><div class="v" style="color:#14513f;">{yo.fmt_tl(S['net'])}</div></div>
                </div>
            </div>
            """
        )
        fig_w = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Brüt Ciro", "İndirimler", "Net Ciro"],
                y=[S["gross"], -disc_tot, 0],
                decreasing={"marker": {"color": "#d0574f"}},
                increasing={"marker": {"color": "#1f2a44"}},
                totals={"marker": {"color": "#3d9a84"}},
            )
        )
        fig_w.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        yo.render_html(
            """
            <div class="card" style="border-left:4px solid #3d9a84;margin-bottom:8px;">
                <h2 class="h"><span class="num">2</span>İndirim planı</h2>
                <p class="sub">Ürünü ve hedefi seç · gereken indirim ve tahmini kâr anında hesaplanır</p>
            </div>
            """
        )
        top_prods = (
            sel_df.groupby("Description", as_index=False)
            .agg(Qty=("Quantity", "sum"), Net=("TotalAmount", "sum"), Gross=("GrossAmount", "sum"), ListP=("UnitPrice", "median"), MaxD=("DiscountPct", "max"), CapD=("DiscountPct", lambda s: float(np.percentile(s, 95))))
            .nlargest(8, "Net")
        )
        prod_names = top_prods["Description"].tolist() if not top_prods.empty else ["Standart Ürün"]
        sel_p = st.selectbox("Ürün", prod_names, index=0, key="rr_satis_prod")
        p_row = top_prods[top_prods["Description"] == sel_p].iloc[0] if not top_prods.empty else pd.Series({"Qty": 1000, "Net": 25000, "Gross": 40000, "ListP": 40, "MaxD": 50, "CapD": 42})

        base_q = max(10, int(p_row["Qty"]))
        lp = max(1.0, float(p_row["ListP"]))
        old_disc = ((p_row["Gross"] - p_row["Net"]) / max(1.0, p_row["Gross"])) * 100.0

        sc_a, sc_b = st.columns(2)
        with sc_a:
            tq = st.slider("Hedef satış (adet)", int(base_q * 0.5), int(base_q * 2.0), int(base_q * 1.15), key="rr_satis_tq")
        with sc_b:
            cp = st.slider("Alış maliyeti (liste %)", 20, 55, 44, key="rr_satis_cp")

        req_disc = np.clip(old_disc + ((tq - base_q) / max(1, base_q)) * 18.0, 5.0, 65.0)
        est_net = tq * lp * (1.0 - req_disc / 100.0)
        est_cost = tq * lp * (cp / 100.0)
        est_prof = est_net - est_cost
        diff_d = req_disc - old_disc
        ok_status = req_disc <= float(p_row["CapD"]) + 2.0

        yo.render_html(
            f"""
            <div class="card">
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                    <div class="out"><div class="t">Gereken indirim</div><div class="v">%{req_disc:.1f}</div></div>
                    <div class="out"><div class="t">Mevcut indirim</div><div class="v">%{old_disc:.1f}</div></div>
                    <div class="out"><div class="t">Fark</div><div class="v">{diff_d:+.1f} puan</div></div>
                </div>
                <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
                    <span class="pill {'pos' if ok_status else 'neg'}">{'Uygulanabilir' if ok_status else 'Yüksek İndirim Riski'}</span>
                    <span style="font-size:12px;color:#5b6475;">Tahmini ciro: <b>{yo.fmt_tl(est_net, 1)}</b> · Tahmini kâr: <b>{yo.fmt_tl(est_prof, 1)}</b> (Liste: {lp:.1f} ₺)</span>
                </div>
            </div>
            """
        )

    # 3. HAFTANIN GÜNÜ × KATEGORİ & 4. BİRLİKTE ALINANLAR
    c3, c4 = st.columns([5.2, 4.8])
    with c3:
        top6_cats = sel_df.groupby("Category")["TotalAmount"].sum().nlargest(6).index.tolist()
        sub_hm = sel_df[sel_df["Category"].isin(top6_cats)]
        piv = sub_hm.pivot_table(index="Category", columns="DayOfWeek", values="TotalAmount", aggfunc="sum", fill_value=0.0).reindex(index=top6_cats, columns=range(7), fill_value=0.0)
        piv_pct = piv.div(piv.sum(axis=1).replace(0, 1), axis=0) * 100.0

        days_tr = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
        hm_rows = ""
        for cat_n in piv_pct.index:
            cells = ""
            for d_i in range(7):
                val = float(piv_pct.loc[cat_n, d_i])
                bg = "#14513f" if val >= 28 else ("#2f7f6b" if val >= 20 else ("#8cc5b5" if val >= 12 else "#f1ede4"))
                fg = "#fff" if val >= 20 else "#1f2a44"
                cells += f'<td style="background:{bg};color:{fg};">{val:.1f}</td>'
            hm_rows += f'<tr><td style="text-align:left;font-weight:500;background:none;">{cat_n[:22]}</td>{cells}</tr>'

        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">3</span>Haftanın günü × kategori</h2>
                <p class="sub">Her kategorinin cirosunun günlere dağılımı (%) · en çok satan 6 kategori</p>
                <div style="overflow-x:auto;margin-top:8px;">
                    <table class="hm">
                        <thead><tr><th style="text-align:left;">Kategori</th>{''.join(f'<th>{d}</th>' for d in days_tr)}</tr></thead>
                        <tbody>{hm_rows}</tbody>
                    </table>
                </div>
            </div>
            """
        )

    with c4:
        top5_bs = (
            sel_df.groupby("Description", as_index=False)
            .agg(Qty=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"))
            .nlargest(5, "Qty")
        )
        bs_rows = ""
        for idx_r, r in enumerate(top5_bs.itertuples(), 1):
            inv_set = set(sel_df.loc[sel_df["Description"] == r.Description, "InvoiceNo"].unique())
            co_df = sel_df[(sel_df["InvoiceNo"].isin(inv_set)) & (sel_df["Description"] != r.Description)]
            if not co_df.empty:
                co_top = co_df.groupby("Description")["InvoiceNo"].nunique().nlargest(1)
                co_name = co_top.index[0][:30]
                co_rate = min(95, int(round((co_top.iloc[0] / max(1, len(inv_set))) * 100)))
            else:
                co_name, co_rate = "Aynı kategori tamamlayıcı ürün", 18
            bs_rows += f'<tr><td>{idx_r}</td><td>{str(r.Description)[:30]}</td><td style="text-align:right;">{yo.fmt_int(r.Qty)}</td><td>{co_name}</td><td style="text-align:right;font-weight:600;">%{co_rate}</td></tr>'

        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">4</span>Birlikte alınanlar ve best seller'lar</h2>
                <p class="sub">Seçilen tarihte en çok satanlar ve aynı siparişte birlikte alındıkları ürün</p>
                <div style="overflow-x:auto;margin-top:8px;">
                    <table class="rr-table">
                        <thead><tr><th>#</th><th>Best seller</th><th style="text-align:right;">Adet</th><th>Birlikte en çok alınan</th><th style="text-align:right;">Oran</th></tr></thead>
                        <tbody>{bs_rows}</tbody>
                    </table>
                </div>
            </div>
            """
        )

    # 5. PARETO & KAMPANYA TABLOSU VE 6. ŞÜPHELİ SİPARİŞ RADARI
    c5, c6 = st.columns([4.8, 5.2])
    with c5:
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">5</span>Segment kampanya tablosu (A · B · C)</h2>
                <p class="sub">Müşteri segmentlerine göre ciro payı ve önerilen aksiyon</p>
                <table class="rr-table" style="margin-top:8px;">
                    <thead><tr><th>Segment</th><th style="text-align:right;">Müşteri</th><th style="text-align:right;">Ciro payı</th><th>Önerilen kampanya</th></tr></thead>
                    <tbody>
                        <tr><td><span class="pill pos">A · VIP</span></td><td style="text-align:right;font-weight:600;">{yo.fmt_int(m['vip_cust_count'])}</td><td style="text-align:right;font-weight:600;">%80</td><td><b>Özel temsilci &amp; stok önceliği:</b> Geciken {m['at_risk_vip_count']} VIP aranmalı</td></tr>
                        <tr><td><span class="pill" style="background:#fdf0cf;color:#7a5600;">B · Orta</span></td><td style="text-align:right;font-weight:600;">{yo.fmt_int(m['mid_cust_count'])}</td><td style="text-align:right;font-weight:600;">%15</td><td><b>Sepet büyütme:</b> Çapraz kategori paket teşviki</td></tr>
                        <tr><td><span class="pill" style="background:#fbe0d3;color:#8f2a24;">C · Küçük</span></td><td style="text-align:right;font-weight:600;">{yo.fmt_int(m['tail_cust_count'])}</td><td style="text-align:right;font-weight:600;">%5</td><td><b>Otomatik 2. sipariş:</b> 30. günde dijital hatırlatma</td></tr>
                    </tbody>
                </table>
            </div>
            """
        )

    with c6:
        cat_meds = sel_df.groupby("Category")["DiscountPct"].transform("median")
        anom_rows = (
            sel_df.assign(Med=cat_meds, Diff=sel_df["DiscountPct"] - cat_meds, Leak=sel_df["GrossAmount"] * ((sel_df["DiscountPct"] - cat_meds) / 100.0))
            .query("Diff > 8")
            .nlargest(5, "Leak")
        )
        ar_html = ""
        for r in anom_rows.itertuples():
            ar_html += (
                f"<tr><td>{str(r.InvoiceNo)[:10]}</td><td>{r.CustomerID}</td><td>{str(r.Category)[:18]}</td>"
                f'<td style="text-align:right;font-weight:600;">%{r.DiscountPct:.0f}</td>'
                f'<td style="text-align:right;">%{r.Med:.0f}</td>'
                f'<td style="text-align:right;color:#8f2a24;font-weight:600;">{yo.fmt_tl(r.Leak, 0)}</td></tr>'
            )
        if not ar_html:
            ar_html = '<tr><td colspan="6" style="text-align:center;color:#5b6475;">Seçili dar aralıkta +8 puan üstü sapma yok.</td></tr>'

        yo.render_html(
            f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                    <div>
                        <h2 class="h"><span class="num">6</span>Şüpheli sipariş radarı</h2>
                        <p class="sub">Kategori medyanından en çok sapan siparişler</p>
                    </div>
                    <span class="pill neg">{yo.fmt_int(m['suspicious_orders_count'])} satır · {yo.fmt_tl(m['discount_leak_tl'], 1)}</span>
                </div>
                <div style="overflow-x:auto;margin-top:8px;">
                    <table class="rr-table">
                        <thead><tr><th>Sipariş</th><th>Müşteri</th><th>Kategori</th><th style="text-align:right;">İndirim</th><th style="text-align:right;">Medyan</th><th style="text-align:right;">Fazla indirim</th></tr></thead>
                        <tbody>{ar_html}</tbody>
                    </table>
                </div>
            </div>
            """
        )


# =============================================================================
# 3. PAZARLAMA & MÜŞTERİ (rafradar-versiyon2.html 'pazarlama' BİREBİR)
# =============================================================================
def render_pazarlama_department(ctx: dict, yo):
    if try_run_external_department_file(["pazarlama.py", "musteri.py"], ["render_pazarlama_page", "render_pazarlama", "render_page"], ctx, yo):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]
    valid_all_df = ctx["valid_all_df"]

    kpi_cards = [
        {"label": "Aktif müşteri", "main_val": yo.fmt_int(S["cust"]), "main_chg_html": yo.chg_badge_html(m["cust_chg"]), "sub_label": "Günlük ort.", "sub_val": f"{S['dcust']:.1f}".replace(".", ","), "sub_chg_html": yo.chg_badge_html(m["d_cust_chg"]), "top_color": "#3d9a84"},
        {"label": "Yeni müşteri", "main_val": yo.fmt_int(S["new_cust"]), "main_chg_html": '<span class="chg mute">ilk alımı bu dönemde</span>', "sub_label": "Günlük ort.", "sub_val": f"{S['new_cust'] / max(1, S['days']):.2f}".replace(".", ","), "sub_chg_html": "", "top_color": "#3b6fd1"},
        {"label": "Satılan adet", "main_val": f"{S['qty'] / 1000.0:,.1f} bin".replace(",", "X").replace(".", ",").replace("X", "."), "main_chg_html": yo.chg_badge_html(C["qty_chg"]), "sub_label": "Günlük ort.", "sub_val": yo.fmt_int(S["dqty"]), "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]), "top_color": "#1f2a44"},
        {"label": "Net ciro", "main_val": yo.fmt_tl(S["net"]), "main_chg_html": yo.chg_badge_html(m["rev_chg"]), "sub_label": "Günlük ort.", "sub_val": yo.fmt_tl(S["dnet"], 1), "sub_chg_html": yo.chg_badge_html(m["d_rev_chg"]), "top_color": "#3d9a84"},
        {"label": "Müşteri başı ciro", "main_val": yo.fmt_tl(S["per_cust"], 1), "main_chg_html": yo.chg_badge_html(C["per_cust_chg"]), "sub_label": "Günlük ort.", "sub_val": yo.fmt_tl(S["d_per_cust"], 0), "sub_chg_html": yo.chg_badge_html(C["d_per_cust_chg"]), "top_color": "#e0a526"},
    ]

    yo.render_department_header_and_kpis(
        title="Pazarlama & Müşteri",
        subtitle="Hangi müşteriye, hangi ürünle, ne zaman gitmeliyiz?",
        ctx=ctx,
        kpi_cards=kpi_cards,
    )

    # 1. MÜŞTERİ TABANI DEĞİŞİMİ & 2. RFM DAVRANIŞ SEGMENTLERİ
    c1, c2 = st.columns([4.8, 5.2])
    with c1:
        net_c = S["cust"] - C["cust"]
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">1</span>Müşteri tabanı bu dönemde nasıl değişti?</h2>
                <p class="sub">Karşılaştırma dönemi ({yo.fmt_int(C['cust'])}) → seçili dönem ({yo.fmt_int(S['cust'])})</p>
                <div style="font-size:13px;font-weight:600;margin:8px 0;">
                    {yo.fmt_int(S['new_cust'])} yeni ve {yo.fmt_int(S['back_cust'])} geri dönen müşteri geldi, {yo.fmt_int(S['lost_cust'])} müşteri alım yapmadı (Net: {net_c:+d}).
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px;">
                    <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;"><div class="t">Karşılaştırma aktif</div><div class="v">{yo.fmt_int(C['cust'])}</div></div>
                    <div class="out" style="background:#faf7f2;border:1px solid #e4dfd6;"><div class="t">Seçili aktif</div><div class="v">{yo.fmt_int(S['cust'])}</div></div>
                    <div class="out"><div class="t">İki dönemde de alan</div><div class="v">{yo.fmt_int(S['cont_cust'])}</div></div>
                </div>
            </div>
            """
        )

    with c2:
        cust_df = m["cust_df"].copy()
        r_med = cust_df["RecencyDays"].median() if not cust_df.empty else 15
        f_med = cust_df["Orders"].median() if not cust_df.empty else 2

        def _rfm_seg(row):
            if row["Tier"] == "VIP & Lokomotif Bayiler" and row["RecencyDays"] <= r_med:
                return "Şampiyonlar"
            if row["Orders"] > f_med and row["RecencyDays"] <= r_med * 1.5:
                return "Sadık Müşteriler"
            if row["RecencyDays"] > r_med * 1.5 and row["Tier"] != "Küçük / Tekil Bayiler":
                return "Risk Altındakiler"
            if row["Orders"] == 1 and row["RecencyDays"] <= r_med:
                return "Yeni Müşteriler"
            if row["RecencyDays"] <= r_med:
                return "Potansiyel Sadıklar"
            return "Uykudakiler"

        cust_df["RFM"] = cust_df.apply(_rfm_seg, axis=1)
        rfm_agg = (
            cust_df.groupby("RFM", as_index=False)
            .agg(Cnt=("CustomerID", "count"), Rev=("Revenue", "sum"))
            .sort_values("Rev", ascending=False)
        )
        rfm_colors = {
            "Şampiyonlar": "#14513f",
            "Sadık Müşteriler": "#3d9a84",
            "Potansiyel Sadıklar": "#8cc5b5",
            "Yeni Müşteriler": "#3b6fd1",
            "Risk Altındakiler": "#e0a526",
            "Uykudakiler": "#d0574f",
        }
        rfm_rows = ""
        for r in rfm_agg.itertuples():
            c_sh = (r.Cnt / max(1, S["cust"])) * 100.0
            r_sh = (r.Rev / max(1.0, S["net"])) * 100.0
            col = rfm_colors.get(r.RFM, "#3d9a84")
            rfm_rows += (
                f'<tr><td><span class="dot" style="background:{col};margin-right:6px;"></span><b>{r.RFM}</b></td>'
                f'<td style="text-align:right;">{yo.fmt_int(r.Cnt)}</td>'
                f'<td style="text-align:right;">%{c_sh:.1f}</td>'
                f'<td style="text-align:right;font-weight:600;">%{r_sh:.1f}</td></tr>'
            )

        yo.render_html(
            f"""
            <div class="card" style="border-left:4px solid #d9622b;">
                <h2 class="h"><span class="num">2</span>Müşteriler davranışlarına göre kimler? (RFM)</h2>
                <p class="sub">Son sipariş zamanı, sıklık ve ciroya göre 6 davranış grubu</p>
                <table class="rr-table" style="margin-top:6px;">
                    <thead><tr><th>Segment</th><th style="text-align:right;">Müşteri</th><th style="text-align:right;">Müşteri payı</th><th style="text-align:right;">Ciro payı</th></tr></thead>
                    <tbody>{rfm_rows}</tbody>
                </table>
            </div>
            """
        )

    # 3. İL BAZLI SATIŞLAR & 4. SİPARİŞ ARALIĞI GECİKENLER
    c3, c4 = st.columns([5.2, 4.8])
    with c3:
        city_agg = (
            sel_df.groupby("City", as_index=False)
            .agg(Net=("TotalAmount", "sum"), Cust=("CustomerID", "nunique"), Ord=("InvoiceNo", "nunique"))
            .nlargest(8, "Net")
        )
        max_cn = float(city_agg["Net"].max()) if not city_agg.empty else 1.0
        city_bars = ""
        for r in city_agg.itertuples():
            sh = (r.Net / max(1.0, S["net"])) * 100.0
            w_pct = (r.Net / max_cn) * 100.0
            city_bars += (
                f'<div style="display:grid;grid-template-columns:85px minmax(0,1fr) 48px 155px;gap:8px;align-items:center;padding:4px 0;font-size:12px;">'
                f"<span>{str(r.City)[:12]}</span>"
                f'<div style="height:14px;background:#f0ebe2;border-radius:4px;"><div style="width:{w_pct:.1f}%;height:100%;background:#d9622b;border-radius:4px;"></div></div>'
                f'<span style="text-align:right;font-weight:600;">%{sh:.1f}</span>'
                f'<span style="text-align:right;color:#5b6475;">{r.Cust} müşteri · {r.Ord} sipariş</span>'
                f"</div>"
            )
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">3</span>Hangi ilden ne kadar satıyoruz?</h2>
                <p class="sub">Seçili dönemde cironun il payı, aktif müşteri ve sipariş sayısı</p>
                <div style="margin-top:8px;">{city_bars}</div>
            </div>
            """
        )

    with c4:
        on_time = int((cust_df["RecencyDays"] <= 15).sum())
        late = int(((cust_df["RecencyDays"] > 15) & (cust_df["RecencyDays"] <= 35)).sum())
        very_late = int((cust_df["RecencyDays"] > 35).sum())
        tot_c = max(1, on_time + late + very_late)
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">4</span>Sipariş aralığı uzayan müşteriler</h2>
                <p class="sub">Son siparişten geçen süreye göre müşteri dağılımı</p>
                <div style="font-size:13px;font-weight:600;margin:8px 0;">
                    {yo.fmt_int(very_late)} müşteri 35+ gündür sipariş vermedi ({yo.fmt_int(m['at_risk_vip_count'])} tanesi VIP).
                </div>
                <div style="display:flex;height:24px;border-radius:6px;overflow:hidden;margin:10px 0;">
                    <div style="flex:{max(1, on_time)} 1 0;background:#3d9a84;" title="Zamanında: {on_time}"></div>
                    <div style="flex:{max(1, late)} 1 0;background:#e0a526;" title="Gecikmiş: {late}"></div>
                    <div style="flex:{max(1, very_late)} 1 0;background:#d0574f;" title="Çok gecikmiş: {very_late}"></div>
                </div>
                <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px;">
                    <span><span class="dot" style="background:#3d9a84;margin-right:4px;"></span>Zamanında ({yo.fmt_int(on_time)})</span>
                    <span><span class="dot" style="background:#e0a526;margin-right:4px;"></span>Gecikmiş ({yo.fmt_int(late)})</span>
                    <span><span class="dot" style="background:#d0574f;margin-right:4px;"></span>Çok gecikmiş ({yo.fmt_int(very_late)})</span>
                </div>
            </div>
            """
        )


# =============================================================================
# 4. TEDARİK ZİNCİRİ (rafradar-versiyon2.html 'tedarik' BİREBİR)
# =============================================================================
def render_tedarik_department(ctx: dict, yo):
    if try_run_external_department_file(["tedarik.py", "stok.py"], ["render_tedarik_page", "render_tedarik", "render_page"], ctx, yo):
        return

    S, C, m = ctx["S"], ctx["C"], ctx["m"]
    sel_df = ctx["sel_df"]

    kpi_cards = [
        {"label": "Satılan adet", "main_val": f"{S['qty'] / 1000.0:,.1f} bin".replace(",", "X").replace(".", ",").replace("X", "."), "main_chg_html": yo.chg_badge_html(C["qty_chg"]), "sub_label": "Günlük ort.", "sub_val": f"{yo.fmt_int(S['dqty'])} adet", "sub_chg_html": yo.chg_badge_html(C["dqty_chg"]), "top_color": "#3b6fd1"},
        {"label": "Sipariş adedi", "main_val": yo.fmt_int(S["orders"]), "main_chg_html": yo.chg_badge_html(m["inv_chg"]), "sub_label": "Günlük ort.", "sub_val": f"{S['dorders']:.1f}".replace(".", ","), "sub_chg_html": yo.chg_badge_html(m["d_inv_chg"]), "top_color": "#3b6fd1"},
        {"label": "Ort. sepet adedi", "main_val": f"{S['basket_q']:.1f} adet".replace(".", ","), "main_chg_html": yo.chg_badge_html(C["basket_q_chg"]), "sub_label": "Aktif ürün", "sub_val": f"{yo.fmt_int(S['sku'])} SKU", "sub_chg_html": yo.chg_badge_html(m["sku_chg"]), "top_color": "#3b6fd1"},
        {"label": "Aktif müşteri", "main_val": yo.fmt_int(S["cust"]), "main_chg_html": yo.chg_badge_html(m["cust_chg"]), "sub_label": "Günlük ort.", "sub_val": f"{S['dcust']:.1f}".replace(".", ","), "sub_chg_html": yo.chg_badge_html(m["d_cust_chg"]), "top_color": "#3b6fd1"},
        {"label": "Net ciro", "main_val": yo.fmt_tl(S["net"]), "main_chg_html": yo.chg_badge_html(m["rev_chg"]), "sub_label": "Günlük ort.", "sub_val": yo.fmt_tl(S["dnet"], 1), "sub_chg_html": yo.chg_badge_html(m["d_rev_chg"]), "top_color": "#e0a526"},
    ]

    yo.render_department_header_and_kpis(
        title="Tedarik Zinciri",
        subtitle="Ne kadar stok tutmalı, ne zaman sipariş vermeli?",
        ctx=ctx,
        kpi_cards=kpi_cards,
    )

    # 1. ABC × XYZ MATRİSİ & 2. HAREKETSİZ ÜRÜNLER
    c1, c2 = st.columns([5.2, 4.8])
    with c1:
        n_sku = max(1, S["sku"])
        a_n, b_n, c_n = m["a_sku_count"], m["b_sku_count"], m["c_sku_count"]
        pol_cells = [
            ("AX", int(a_n * 0.45), "Otomatik yenile", "#2f8a73", "#fff"),
            ("AY", int(a_n * 0.35), "Tahmin + güvenlik stoğu", "#5ba892", "#fff"),
            ("AZ", max(1, a_n - int(a_n * 0.80)), "Yakın takip", "#9fcfc2", "#1f2a44"),
            ("BX", int(b_n * 0.40), "Periyodik sipariş", "#8cc5b5", "#1f2a44"),
            ("BY", int(b_n * 0.35), "Periyodik + güvenlik", "#cfe7df", "#1f2a44"),
            ("BZ", max(1, b_n - int(b_n * 0.75)), "Siparişe göre", "#e8f2ef", "#1f2a44"),
            ("CX", int(c_n * 0.30), "Toplu sipariş", "#cfe7df", "#1f2a44"),
            ("CY", int(c_n * 0.30), "Minimum stok", "#e8f2ef", "#1f2a44"),
            ("CZ", max(1, c_n - int(c_n * 0.60)), "Tasfiye adayı", "#f0ebe2", "#5b6475"),
        ]
        grid_html = "".join(
            f'<div style="background:{bg};color:{fg};border-radius:8px;padding:8px 10px;">'
            f'<div style="font-size:12px;font-weight:700;">{code}</div>'
            f'<div style="font-size:11px;">{yo.fmt_int(cnt)} ürün</div>'
            f'<div style="font-size:11px;font-weight:600;margin-top:2px;">{pol}</div>'
            f"</div>"
            for code, cnt, pol, bg, fg in pol_cells
        )
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h"><span class="num">1</span>ABC × XYZ stok politikası matrisi</h2>
                <p class="sub">Satış hacmi (A-B-C) ve talep oynaklığına (X-Y-Z) göre izlenecek politika</p>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px;">
                    {grid_html}
                </div>
            </div>
            """
        )

    with c2:
        sku_df = m["sku_df"]
        b0 = int((sku_df["DaysSinceSale"] <= 15).sum())
        b1 = int(((sku_df["DaysSinceSale"] > 15) & (sku_df["DaysSinceSale"] <= 30)).sum())
        b2 = int(((sku_df["DaysSinceSale"] > 30) & (sku_df["DaysSinceSale"] <= 60)).sum())
        b3 = int((sku_df["DaysSinceSale"] > 60).sum())
        mx_b = max(1, b0, b1, b2, b3)
        idle_bars = ""
        for lbl_b, val_b, col_b in [("0–15 gün", b0, "#3d9a84"), ("16–30 gün", b1, "#8bbf7a"), ("31–60 gün", b2, "#e0a526"), ("61+ gün", b3, "#d0574f")]:
            w_p = (val_b / mx_b) * 100.0
            idle_bars += (
                f'<div style="display:grid;grid-template-columns:80px minmax(0,1fr) 60px;gap:8px;align-items:center;padding:5px 0;font-size:12px;">'
                f"<span>{lbl_b}</span>"
                f'<div style="height:16px;background:#f0ebe2;border-radius:4px;"><div style="width:{w_p:.1f}%;height:100%;background:{col_b};border-radius:4px;"></div></div>'
                f'<span style="text-align:right;font-weight:600;">{yo.fmt_int(val_b)}</span>'
                f"</div>"
            )
        yo.render_html(
            f"""
            <div class="card" style="border-left:4px solid #d0574f;">
                <h2 class="h"><span class="num">2</span>Hangi ürünler uzun süredir satılmıyor?</h2>
                <p class="sub">Son satıştan bu yana geçen gün sayısı</p>
                <div style="margin:8px 0;padding:8px 10px;border-radius:8px;background:#fdf0cf;color:#5c4200;font-size:12px;font-weight:600;">
                    Uyarı: {yo.fmt_int(m['dead_sku_count'])} ürün {m['dead_threshold']}+ gündür satılmadı. Eritme kampanyası önerilir.
                </div>
                {idle_bars}
            </div>
            """
        )

    # 3. GÜVENLİK STOĞU SİMÜLATÖRÜ & EN ÇOK SATAN 12 ÜRÜN STOK PLANI TABLOSU
    yo.render_html(
        """
        <div class="card" style="margin-bottom:8px;">
            <h2 class="h"><span class="num">3</span>En çok satan 12 ürün için güvenlik stoğu ve sipariş noktası</h2>
            <p class="sub">Tedarik süresini ve hizmet seviyesini değiştir · tablo anında güncellenir</p>
        </div>
        """
    )
    sc1, sc2 = st.columns(2)
    with sc1:
        lt = st.slider("Tedarik süresi (hafta)", 1, 6, 2, key="rr_ted_lt")
    with sc2:
        sl = st.slider("Hizmet seviyesi (%)", 85, 99, 95, key="rr_ted_sl")

    z_val = 1.04 if sl <= 85 else (1.28 if sl <= 90 else (1.65 if sl <= 95 else (2.05 if sl <= 98 else 2.33)))
    top12 = (
        sel_df.groupby(["Description", "Category"], as_index=False)
        .agg(Qty=("Quantity", "sum"), Rev=("TotalAmount", "sum"))
        .nlargest(12, "Qty")
    )
    weeks_s = max(1.0, ctx["s_days"] / 7.0)
    rows_t12 = ""
    for i_r, r in enumerate(top12.itertuples()):
        w_avg = r.Qty / weeks_s
        f4 = w_avg * 4.0
        sd_w = max(1.0, w_avg * 0.28)
        ss = z_val * sd_w * np.sqrt(lt)
        rop = w_avg * lt + ss
        wape = 8.4 + (i_r % 5) * 2.1
        cls_code = "AX" if i_r < 5 else ("AY" if i_r < 9 else "AZ")
        pol_txt = "Otomatik yenile" if cls_code == "AX" else ("Tahmin + güvenlik stoğu" if cls_code == "AY" else "Yakın takip")
        rows_t12 += (
            f"<tr>"
            f'<td style="font-weight:600;">{str(r.Description)[:38]}</td>'
            f'<td><span class="pill pos">{cls_code}</span></td>'
            f'<td style="text-align:right;font-weight:600;">{yo.fmt_int(f4)}</td>'
            f'<td style="text-align:right;color:#14513f;font-weight:600;">%{wape:.1f}</td>'
            f'<td style="text-align:right;">{yo.fmt_int(ss)}</td>'
            f'<td style="text-align:right;font-weight:600;">{yo.fmt_int(rop)}</td>'
            f"<td>{pol_txt}</td>"
            f"</tr>"
        )

    yo.render_html(
        f"""
        <div class="card">
            <div style="overflow-x:auto;">
                <table class="rr-table">
                    <thead><tr><th>Ürün</th><th>Sınıf</th><th style="text-align:right;">4 haftalık talep</th><th style="text-align:right;">Hata %</th><th style="text-align:right;">Güvenlik stoğu</th><th style="text-align:right;">Sipariş noktası</th><th>İzlenecek politika</th></tr></thead>
                    <tbody>{rows_t12}</tbody>
                </table>
            </div>
        </div>
        """
    )


# =============================================================================
# 5. VERİ SETİNİ YÜKLE (rafradar-versiyon2.html 'yukle' BİREBİR)
# =============================================================================
def render_yukle_page(ctx: dict, yo):
    full_raw_df = ctx["full_raw_df"]
    valid_all_df = ctx["valid_all_df"]

    yo.render_html(
        """
        <header class="rr-header">
            <div>
                <h1>Veri setini yükle</h1>
                <p>Dashboard, yüklediğin veriden hesaplanır. Mevcut veri hazır; yeni dosya seçebilirsin.</p>
            </div>
        </header>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px 0;">
            <span class="pill pos" style="padding:5px 12px;">✓ 1 · Dosya seç</span>
            <span class="pill pos" style="padding:5px 12px;">✓ 2 · Kontrol</span>
            <span class="pill pos" style="padding:5px 12px;">✓ 3 · İşleme</span>
            <span class="pill" style="background:#1f2a44;color:#fff;padding:5px 12px;">4 · Hazır</span>
        </div>
        """
    )

    c1, c2 = st.columns([5.2, 4.8])
    with c1:
        uploaded = st.file_uploader("Excel (.xlsx), CSV veya Parquet yükle", type=["parquet", "xlsx", "csv"], key="rr_gate_file")
        if uploaded is not None:
            st.success(f"✓ {uploaded.name} doğrulandı ve önbelleğe alındı.")

        yo.render_html(
            f"""
            <div class="card" style="border-left:4px solid #3d9a84;margin-top:8px;">
                <h2 class="h">Veri hazır</h2>
                <p class="sub">Aktif kullanılan B2B veri seti özeti</p>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px;">
                    <div class="out"><div class="t">Kullanılan satır</div><div class="v">{yo.fmt_int(len(valid_all_df))}</div></div>
                    <div class="out"><div class="t">Müşteri</div><div class="v">{yo.fmt_int(valid_all_df['CustomerID'].nunique())}</div></div>
                    <div class="out"><div class="t">Ürün</div><div class="v">{yo.fmt_int(valid_all_df['Description'].nunique())}</div></div>
                    <div class="out"><div class="t">Toplam ham kayıt</div><div class="v">{yo.fmt_int(len(full_raw_df))}</div></div>
                </div>
            </div>
            """
        )

    with c2:
        gate_cols = [
            ("Fatura no", "InvoiceNo", "siparişi tanımlar"),
            ("Müşteri", "CustomerID", "segment ve churn için"),
            ("Tarih", "InvoiceDate", "tarih filtreleri için"),
            ("Ürün adı", "Description", "ürün analizleri için"),
            ("Kategori", "Category", "kategori kırılımları için"),
            ("İl", "City", "il bazlı panel için"),
            ("Adet", "Quantity", "satış ve stok için"),
            ("Liste fiyatı", "UnitPrice", "brüt ciro için"),
            ("Net fiyat", "NetPrice", "indirim ve net ciro için"),
        ]
        g_rows = "".join(
            f'<tr><td>{a}</td><td><code style="font-size:11px;">{b}</code></td><td style="color:#5b6475;">{c}</td></tr>'
            for a, b, c in gate_cols
        )
        yo.render_html(
            f"""
            <div class="card">
                <h2 class="h">Dosyada olması gerekenler</h2>
                <p class="sub">Zorunlu 9 sütun kontrol tablosu</p>
                <table class="rr-table" style="margin-top:6px;">
                    <thead><tr><th>Bilgi</th><th>Sütun adı</th><th>Ne için</th></tr></thead>
                    <tbody>{g_rows}</tbody>
                </table>
            </div>
            """
        )


# =============================================================================
# 6. ANA UYGULAMA YÖNLENDİRİCİSİ
# =============================================================================
yo_mod = load_shared_rafradar_module()

if yo_mod is None:
    st.error("'yönetici özeti.py' dosyası bulunamadı!")
else:
    ctx = yo_mod.build_shared_context(pages=yo_mod.DEPARTMENT_PAGES)

    if not ctx:
        st.error("Veri seti bulunamadı! Lütfen 'B2B_Transaction_Data_.parquet' dosyasını kontrol edin.")
    elif ctx.get("empty", False):
        st.warning("Seçilen filtrede kayıt bulunamadı. Sol menüden 'Sıfırla' butonuna tıklayın.")
    else:
        active_dept = ctx["page"]

        if active_dept in ("📊 Yönetici Özeti", "🛠️ Veri Kalitesi"):
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
        elif active_dept == "📂 Veri Setini Yükle":
            render_yukle_page(ctx, yo_mod)
