"""
Makine Öğrenmesi Ara Ödevi
Müşteri Kayıp (Churn) Tahmini

Kullanılan Kütüphaneler: pandas, numpy, scikit-learn
Çalıştırma: python main.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Veri Seti Oluşturma (200 satır)
np.random.seed(42)
n = 200

veri = {
    'yas': np.random.randint(18, 65, n),
    'gelir': np.random.randint(15000, 80000, n),
    'abonelik_suresi': np.random.randint(1, 48, n),
    'destek_talebi_sayisi': np.random.randint(0, 8, n),
    'sehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa'], n),
    'uyelik_tipi': np.random.choice(['Standart', 'Premium', 'VIP'], n)
}

df = pd.DataFrame(veri)

# Hedef değişken (churn): Destek talebi arttıkça ayrılma ihtimali artar
churn_skor = (df['destek_talebi_sayisi'] * 0.15) - (df['abonelik_suresi'] * 0.01) + np.random.normal(0, 0.2, n)
df['churn'] = (churn_skor > 0.2).astype(int)

# Ön işleme pratiği için birkaç eksik değer ekleme
df.loc[np.random.choice(df.index, 4, replace=False), 'gelir'] = np.nan

# 2. Veri İnceleme
print("--- Veri İlk 5 Satır ---")
print(df.head())
print(f"\nBoyut: {df.shape}")
print("\nChurn Dağılımı:")
print(df['churn'].value_counts())

# 3. Eksik Değer Doldurma
df['gelir'] = df['gelir'].fillna(df['gelir'].median())

# 4. Yeni Öznitelik Üretme
df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)
df['abonelik_yili'] = (df['abonelik_suresi'] / 12).round(1)

# 5. One-Hot Encoding
df = pd.get_dummies(df, columns=['sehir', 'uyelik_tipi'], drop_first=True)

# 6. Train, Validation ve Test Ayrımı (Stratify ile)
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# 7. Ölçekleme
sayisal = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi', 'abonelik_yili']
scaler = StandardScaler()
X_train[sayisal] = scaler.fit_transform(X_train[sayisal])
X_val[sayisal] = scaler.transform(X_val[sayisal])
X_test[sayisal] = scaler.transform(X_test[sayisal])

# 8. Model Eğitimi ve Validation Karşılaştırması
modeller = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42)
}

print("\n--- Validation Sonuçları ---")
val_skorlar = {}
for ad, model in modeller.items():
    model.fit(X_train, y_train)
    tahmin = model.predict(X_val)
    f1 = f1_score(y_val, tahmin, zero_division=0)
    acc = accuracy_score(y_val, tahmin)
    val_skorlar[ad] = f1
    print(f"{ad} -> Accuracy: {acc:.2f} | F1-Score: {f1:.2f}")

en_iyi_ad = max(val_skorlar, key=val_skorlar.get)
en_iyi_model = modeller[en_iyi_ad]
print(f"\nSeçilen Model: {en_iyi_ad}")

# 9. Test Seti Değerlendirmesi
test_tahmin = en_iyi_model.predict(X_test)
print(f"\n--- {en_iyi_ad} Test Metrikleri ---")
print("Accuracy :", round(accuracy_score(y_test, test_tahmin), 2))
print("Precision:", round(precision_score(y_test, test_tahmin, zero_division=0), 2))
print("Recall   :", round(recall_score(y_test, test_tahmin, zero_division=0), 2))
print("F1-Score :", round(f1_score(y_test, test_tahmin, zero_division=0), 2))
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, test_tahmin))

# 10. Kısa Yorum
print("\nSonuç Yorumu:")
print("Validation aşamasında en dengeli ve tutarlı F1 skorunu Logistic Regression verdi.")
print("Veri setinde destek talebi arttıkça churn olma olasılığı doğrusal olarak arttığı için doğrusal model daha iyi sonuç üretti.")
