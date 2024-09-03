# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:22:47 2024

@author: MUSTAFA
"""

# Görevler
# 1- Veri Setini İndirme ve Yükleme: UCI Machine Learning Repository’den Iris veri 
#setini indir.Python kullanarak veri setini yükle ve ilk birkaç satırı görüntüle.

import pandas as pd 
pd.set_option('Display.max_columns', 500)
pd.set_option('Display.width', None)
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
# Veri setini yükleme
iris = load_iris()
iris_df= pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['species'] = iris.target

# İlk satırları görmek.
iris_df.head()

# Veri Temizleme: 1- Eksik verileri tespit et ve uygun yöntemlerle doldur veya çıkar.
# 2- Verilerin türlerini kontrol et ve gerektiğinde dönüştür.

iris_df.isnull().sum() # Boş verimiz yok.

iris_df.info() # Verilerimiz float türünde

iris_df.describe() 

iris_df['species'] = iris_df['species'].astype('category') # Çiçek kendi içinde 3 
# türe ayrılıyor. 

print('')

# Keşifsel Veri Analizi (EDA)
# 1- Temel istatistikleri hesapla (ortalama, medyan, standart sapma vb.).
# 2- Veri dağılımlarını görselleştir (histogram, boxplot vb.).

#Matematiksel değerleri hesaplamak için bir fonksiyon
def iris_statistic(dataframe, target, categorical_col):
    print('Target: {}, categorical col : {}'.format(target, categorical_col))
    print('')
    print(pd.DataFrame({'TARGET_MEAN' : dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    print('########################################################')
    print(pd.DataFrame({'TARGET_STD' : dataframe.groupby(categorical_col)[target].std()}), end='\n\n\n')
    print('########################################################')
    print(pd.DataFrame({'TARGET_MEDIAN': dataframe.groupby(categorical_col)[target].median()}), end='\n\n\n')
    print('########################################################')
    quantiles = [0.05, 0.1, 0.15, 0.25, 0.35, 0.50, 0.75, 0.85, 0.90, 0.99]
    print('TARGET_QUANTILES',dataframe[target].describe(quantiles).T)
iris_statistic(iris_df, 'sepal length (cm)', 'species')
iris_statistic(iris_df, 'sepal width (cm)', 'species')
iris_statistic(iris_df, 'petal length (cm)', 'species')
iris_statistic(iris_df, 'petal width (cm)', 'species')

# Veri dağılımları Görselleştirme
# Boxplot
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.boxplot(data=iris_df)
plt.show()

# Scatter plot

sns.scatterplot(data = iris_df, x='sepal length (cm)',y='sepal width (cm)', hue='species')
plt.title('Sepal Length (cm) - Sepal Width (cm)')
plt.show()

sns.scatterplot(data = iris_df, x= 'petal length (cm)', y='petal width (cm)', hue = 'species')
plt.title('Petal Length (cm) - Petal Width (cm)')
plt.show()

# Histogram

iris_df.hist(bins= 20, figsize=(10,10))
plt.title('İris Data - Histogram')
plt.show()

# Pairplot

sns.pairplot(iris_df, hue='species')
plt.title('İris Data - Pairplot')

# Violin Plot

plt.figure(figsize=(10,6))
sns.violinplot(x='species', y= 'sepal length (cm)', data = iris_df)

# Korelasyon matrisi ve heatmap

corr_matrix = iris_df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot= True, cmap = 'coolwarm')
plt.title('Corelation')
plt.show()


# Özellik Mühendisliği : 1 - Yeni özellikler oluştur (örneğin, sepal ve petal oranları).
# 2- Özelliklerin önemini değerlendir.
# Sepal - Petal Oranları

iris_df['sepal_ratio'] = iris_df['sepal length (cm)'] / iris_df['sepal width (cm)']
iris_df['petal_ratio'] = iris_df['petal length (cm)'] / iris_df['petal width (cm)']

# Sepal Length uzunluğunu kategorilere ayırma

iris_df['sepal_legnth_category'] = pd.cut(iris_df['sepal length (cm)'], bins = [0, 5.0, 6.5, 8.0],
                                          labels = ['short', 'medium', 'long'])
print(iris_df.head())

species_names = {0 : 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species_names'] = iris_df['species'].map(species_names)

plt.figure(figsize=(10,6))
sns.boxplot(x= 'species', y= 'sepal_ratio', data = iris_df, )
plt.title('Sepal Ratio')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x= 'species', y= 'petal_ratio', data = iris_df, )
plt.title('Petal Ratio')
plt.show()


# Özellikler arasındaki ilişkiyi names üzerinde açıklıyalım

sns.pairplot(iris_df, hue='species_names')
plt.title('Species Relations')
plt.show()

# Makine Öğrenmesi Modeli Oluşturma:
# 1- Veriyi eğitim ve test setlerine ayır.
# 2- Bir veya birden fazla makine öğrenmesi modeli oluştur ve eğit (örneğin, lojistik regresyon, 
#    karar ağaçları, k-en yakın komşu).
# 3- Model performansını değerlendir (doğruluk, F1 skoru vb.).


from sklearn.model_selection import train_test_split

x = iris_df.drop(columns= ['species', 'species_names', 'sepal_legnth_category'])
y = iris_df['species']

# Veriye eğitim ve test serilerine ayırmak

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
print('Eğitim seti boyutu:', x_train.shape)
print('Test seti boyutu:', x_test.shape)

from sklearn.linear_model import LogisticRegression

# Modeli oluşturmak
model = LogisticRegression(max_iter=200)

# Modeli eğitme
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, f1_score, classification_report

#Tahminler

y_pred = model.predict(x_test)

# Doğruluk ve f1 skoru

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Doğruluk:', accuracy)
print('F1 skoru:', f1)
print('\nSınıflandırma Raporu:\n', classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion Matrix görselleştirme

plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=species_names.values(),
            yticklabels=species_names.values())
plt.xlabel('Tahmin edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.show()
