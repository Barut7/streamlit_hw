# B2B Karar Destek Sistemi

Bir B2B kitap dağıtıcısının 2020 yılı işlem verisinden satış, pazarlama ve tedarik zinciri ekipleri için karar destek paneli. Streamlit ile yazıldı, Streamlit Community Cloud'da yayında.

- **Canlı panel:** https://apphw-j5keu4uvmwbod7fnibwrtc.streamlit.app/
- **Ders:** İTÜ ISL 533 Veri Madenciliği, Ödev 1

## Veri

`B2B_Transaction_Data_.parquet` (B2B_Transaction_Data_.xlsx dosyasının aynısı, daha hızlı okunsun diye parquet).
Yaklaşık 664 bin temiz satır, 1.201 müşteri, 41.194 ürün, 2020 yılı, net ciro ≈ 29,6 milyon ₺.

Uygulama açılınca bu dosyayı otomatik yükler. İstersen "Veri yükle" sayfasından kendi dosyanı da yükleyebilirsin (aynı 9 zorunlu sütun gerekir).

## Sayfalar

| Sayfa | Ne gösterir |
|---|---|
| Yönetici Özeti | Üç departmanın toplam ve günlük KPI'ları, yıllık bakış, müşteri/ürün yapısı, departmanlar arası çıkarımlar, 30 günlük aksiyon planı ve etki simülasyonu |
| Satış & Fiyat | Ciro ve indirim analizi, kategori ve şehir kırılımı, fazla indirim tespiti, birlikte alınan ürünler |
| Pazarlama & Müşteri | Pareto (A/B/C), RFM segmentleri, churn riski, kohort elde tutma |
| Tedarik Zinciri | ABC-XYZ sınıfları, talep tahmini, güvenlik stoğu, yavaş hareket eden ürünler |
| Veri Kalitesi | Alan doluluğu, atılan satırlar, metrik tanımları ve hesaplama notları |

Sol menüden tarih aralığı, karşılaştırma aralığı, kategori, ürün ve şehir filtreleri her sayfada çalışır. Sağ alttaki asistan veriyle ilgili soruları yanıtlar.

## Kullanılan yöntemler

- **Veri ön işleme:** zorunlu sütun kontrolü, tip dönüşümü, hatalı/eksik satırların atılması, alan doluluğu raporu
- **Churn tahmini:** Gradient Boosting, AUC ile değerlendirme, müşteri başına beklenen 30 günlük ciro
- **Talep tahmini:** Holt (damped), ortalama ve SBA yöntemleri, WAPE ile geri test
- **Segmentasyon:** RFM, Pareto (A/B/C), ABC-XYZ
- **Kohort analizi:** yeni müşterilerin çeyreklik elde tutulması
- **Fazla indirim:** sipariş indirim oranının kategori medyanıyla karşılaştırılması

## Yerelde çalıştırma

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dosya yapısı

```
streamlit_app.py                  Ana uygulama (veri, analitik, arayüz ve tüm sayfalar)
B2B_Transaction_Data_.parquet     Veri seti
requirements.txt                  Kütüphaneler
.streamlit/config.toml            Tema ayarları
```

## Ekip

Hilal Barut, Selin Özen ve grup arkadaşı. Satış & Fiyat ile Veri yükle sayfaları Selin'e aittir.
