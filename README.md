# Makine Öğrenmesi Ara Ödevi - Churn Tahmini

Bu ödevde müşteri verileri üzerinden müşterinin şirketten ayrılıp ayrılmayacağını (churn) tahmin eden temel bir makine öğrenmesi akışı hazırlanmıştır.

## Yapılan Adımlar
1. Pandas ile örnek müşteri veri seti oluşturuldu.
2. Eksik veriler incelendi ve medyan ile dolduruldu.
3. `destek_talebi_var_mi` ve `abonelik_yili` adında iki yeni öznitelik üretildi.
4. Kategorik değişkenlere One-Hot Encoding, sayısal değişkenlere StandardScaler uygulandı.
5. Veri kümesi %70 Train, %15 Validation ve %15 Test olarak `stratify` ile ayrıldı.
6. Logistic Regression, KNN ve Decision Tree modelleri eğitildi.
7. Validation aşamasında en iyi F1 skorunu veren model seçilip Test seti üzerinde metrikleri hesaplandı.

## Çalıştırma
```bash
pip install -r requirements.txt
python main.py
Sonuç
En iyi performansı Logistic Regression modeli gösterdi. Müşteri terk durumunda destek talebi sayısı doğrudan etkili olduğu için doğrusal model ilişkileri daha iyi yakaladı.
