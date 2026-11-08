"""RafRadar veri okuma, doğrulama, temizlik ve önbellek modülü."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ZORUNLU = [
    "InvoiceNo",
    "CustomerID",
    "InvoiceDate",
    "Description",
    "Category",
    "City",
    "Quantity",
    "UnitPrice",
    "NetPrice",
]

KOK = Path(__file__).resolve().parent.parent


def _dosya_oku(dosya_adi: str, icerik: bytes) -> pd.DataFrame:
    ad = dosya_adi.lower()
    buf = io.BytesIO(icerik)
    if ad.endswith(".parquet"):
        return pd.read_parquet(buf)
    if ad.endswith(".xlsx") or ad.endswith(".xls"):
        return pd.read_excel(buf)
    if ad.endswith(".csv"):
        try:
            return pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="latin1", sep=None, engine="python")
    raise ValueError(f"Desteklenmeyen dosya uzantısı: {dosya_adi}")


@st.cache_data(show_spinner=False)
def hazirla(dosya_adi: str, icerik: bytes) -> tuple[pd.DataFrame | None, int, list[str], dict]:
    """Dosyayı okur, zorunlu sütunları kontrol eder, temizler ve türetilmiş sütunları ekler."""
    hatalar: list[str] = []
    try:
        ham = _dosya_oku(dosya_adi, icerik)
    except Exception as exc:
        return None, 0, [f"Dosya okunamadı ({dosya_adi}): {exc}"], {}

    if ham is None or ham.empty:
        return None, 0, ["Yüklenen dosya boş görünüyor."], {}

    ham.columns = [str(c).strip() for c in ham.columns]
    eksik = [c for c in ZORUNLU if c not in ham.columns]
    if eksik:
        hatalar.append(f"Eksik zorunlu sütunlar: {', '.join(eksik)}. Lütfen 9 zorunlu sütunun tamamını içeren bir dosya yükleyin.")
        return None, 0, hatalar, {}

    ham_satir = len(ham)
    ham_sutun = len(ham.columns)

    # Boşluk ve bilimsel gösterim denetimi
    kontrol_sutunlari = ["InvoiceNo", "CustomerID", "InvoiceDate", "Description", "Category", "City", "StockCode"]
    bos: dict[str, int] = {}
    for k in kontrol_sutunlari:
        if k in ham.columns:
            s = ham[k]
            bos_say = int(s.isna().sum() + (s.astype(str).str.strip().isin(["", "nan", "None", "NULL"])).sum() - s.isna().sum())
            bos[k] = max(0, bos_say)

    bilimsel = 0
    if "StockCode" in ham.columns:
        sc_str = ham["StockCode"].dropna().astype(str).str.strip()
        bilimsel = int(
            (
                sc_str.str.contains("E+", case=False, regex=False)
                | sc_str.str.match(r"^97[89]\d{4,6}00000$")
            ).sum()
        )

    tekrar = int(ham.duplicated().sum())

    # Tip dönüşümleri
    df = ham.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["NetPrice"] = pd.to_numeric(df["NetPrice"], errors="coerce")

    if df["InvoiceDate"].isna().all():
        return None, 0, ["InvoiceDate sütunundaki tarihler okunamadı. Tarih biçimini kontrol edin."], {}

    # Temizlik: sıfır/negatif miktar veya fiyat ya da geçersiz tarih/müşteri satırlarını çıkar
    gecerli = (
        df["InvoiceDate"].notna()
        & df["CustomerID"].notna()
        & df["Description"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (df["NetPrice"] > 0)
        & (df["NetPrice"] <= df["UnitPrice"] * 1.0001)
    )
    df = df.loc[gecerli].copy()
    if df.empty:
        return None, 0, ["Temizlik sonrası geçerli satış satırı kalmadı (tüm miktar veya fiyatlar ≤ 0)."], {}

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Category"] = df["Category"].where(df["Category"].notna(), None)
    df["City"] = df["City"].fillna("Belirsiz").astype(str).str.strip()

    df["gun"] = df["InvoiceDate"].dt.normalize()
    df["dow"] = df["InvoiceDate"].dt.dayofweek
    df["gross"] = df["Quantity"] * df["UnitPrice"]
    df["net"] = df["Quantity"] * df["NetPrice"]
    df["disc"] = np.where(df["gross"] > 0, np.clip(1.0 - df["net"] / df["gross"], -0.5, 0.999), 0.0)

    temiz_satir = len(df)
    atilan = ham_satir - temiz_satir

    kalite = {
        "ham_satir": ham_satir,
        "temiz_satir": temiz_satir,
        "atilan": atilan,
        "ham_sutun": ham_sutun,
        "yeni_sutun": 3,
        "bos": bos,
        "bilimsel": bilimsel,
        "tekrar": tekrar,
    }
    return df, atilan, [], kalite


def ornek_dosyalar() -> list[Path]:
    """Kök dizin ve data/ klasöründeki örnek parquet/xlsx dosyalarını döndürür."""
    adaylar = (
        sorted(KOK.glob("data/*.parquet"))
        + sorted(KOK.glob("*.parquet"))
        + sorted(KOK.glob("data/*.xlsx"))
        + sorted(KOK.glob("*.xlsx"))
    )
    gorulen = set()
    sonuc = []
    for p in adaylar:
        rp = p.resolve()
        if rp not in gorulen:
            gorulen.add(rp)
            sonuc.append(p)
    return sonuc


def veri() -> pd.DataFrame:
    """Session state'teki temiz veriyi döndürür; yoksa örnek dosyadan otomatik yükler."""
    if "df" not in st.session_state:
        ornekler = ornek_dosyalar()
        if ornekler:
            f = ornekler[0]
            df, atilan, hatalar, kalite = hazirla(f.name, f.read_bytes())
            if not hatalar and df is not None:
                st.session_state["df"] = df
                st.session_state["atilan"] = atilan
                st.session_state["kalite"] = kalite
                st.session_state["dosya_adi"] = f.name
    if "df" not in st.session_state:
        st.warning("Önce veri setini yükleyin.")
        st.stop()
    return st.session_state["df"]
