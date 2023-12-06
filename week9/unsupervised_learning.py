################################
# Unsupervised Learning
################################
#İlgilendiğimiz veri setinde bir bağımlı değişken ve ya hedef değişkeni yoksa  yani ilgili gözlem birimlerinde meydana gelen gözlemlere karşılık ortaya ne çıktı bilgisi yoksa, label yoksa gözetimsiz öğrenmedir


#pip install yellowbrick
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
################################
# K-Means
################################
#Amaç gözlemleri birbirlerine olan benzerliklerine göre kümelere ayırmaktır. Sınıflandırma problemi gibi düşünülebilir, birbirine benzer olanları gruplara ayırıyoruz ama sınıflandırma probleminde, önceden sınıflar bellidir. Bu örnekte ise sınıflar belli değildir, benzer olanları bir sınıf yapar.

#Nasıl çalışır?
#Adım1: Küme sayısı belirlenir
#adım2: Rastgele K merkez seçilir
#adım3: her gözlem için k merkezlere uzaklıklar hesaplanır
#adım4: her gözlem en yakın olduğu merkeze yani kümeye atanır.
#adım5: atama işlemlerinden sonra oluşan kümeler için tekrar merkez hesaplamaları yapılır
#adım6: Bu işlem belirlenen bir iterasyon adedince tekrar edilir ve küme içi hata kareler toplamlarının toplamının minimum olduğu durumdaki gözlemlerin kümelenme yapısı nihai kümelenme olarak seçilir.

#özetle: başta rastgele k tane merkez belirlendi, belirlenen k merkeze uzaklıklar hesaplandı ve her gözlem en yakın olduğu merkezlere atandı ve küme oldular. Daha sonra küme içinde tekrar merkez hesabı yapılır. Tekrar gözlem birimlerinin merkezlere olan uzaklıkları üzerinden yeniden küme ataması yapılır ve yeniden merkez hesapları yapılır. yukardaki durum verilen iterasyon kadar  tekrar edilir. kümeler kendi içinde homojen,birbirlerine göre heterojen olsun istiyoruz. Matematiksel bir şekilde  SSE / SSR metrikleriyle belirleriz. Kümenin merkezindeki gözlem biriminin değerleriyle etrafındaki diğerlerinin gözlem birimlerinin değerlerinin farklarının karelerini alıp topladığımızda kümenin içindeki hata karesini buluruz. Hatamızı düşürmek için küme içindeki merkezlerin yerlerini değiştirip, en düşük SSE'ye ulaştığımızda o merkez olarak seçilir.

#reading csv
df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)

#first check at data
df.head()
df.isnull().sum()
df.info()
df.describe().T

#değişkenlerimizi 0 ile 1  arasında dönüştürüyoruz
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

#kmeans algoritmasıyla modelimizi kurup fit ediyoruz
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters #küme sayısı
kmeans.cluster_centers_ #küme merkezleri
kmeans.labels_  #küme etiketleri
kmeans.inertia_    # sum of squared distances

################################
# Optimum Küme Sayısının Belirlenmesi
################################
#farklı k parametre değerlerine göre SSE/SSR/SSD'yi incelemek istiyoruz
kmeans = KMeans()
ssd = []
K = range(1, 30)
#döngü aracılığıyla 30'a kadar deniyoruz ve bütün intertia değerlerini görüyoruz
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

#görselleştirip bakıyoruz
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()
#küme sayısı arttıkça ssr düşmüş gözüküyor

#elbow yöntemine göre 9'i optimum nokta olarak belirlemiş
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
#value ile bakabiliriz
elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################
#modelimizi verdiği best value'ya göre kuruyoruz
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

#labelları alıyoruz
clusters_kmeans = kmeans.labels_
#df'imizi tekrardan okutuyoruz
df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)
#yeni bir degisken ekledik
df["cluster"] = clusters_kmeans
df.head()

#0 numaralı clusterdan kurtulmak için yapıyoruz
df["cluster"] = df["cluster"]+ 1

#5 numaralı clusterda hangi eyaletler  var bakiyoruz
df[df["cluster"]==5]
#groupby'a alıp bakabiliyoruz hepsine tek tek
df.groupby("cluster").agg(["count","mean","median"])
#kaydetmek içinde  to_Csv kullanıyoruz
df.to_csv("clusters.csv")

################################
# Hierarchical Clustering
################################
# Gözlemleri birbirlerine olan benzerliklerine göre alt kümelere ayırmak amacımız. Fakat kümelere ayırma işlemi Hiyerarşik olarak gerçekleştiriliyor. Birleştirici ve Bölümleyici olarak iki genel başlığımız var. Birleştirici yöntemde çalışmanın başında bütün gözlem birimleri tek bir küme gibi düşünülüp yukarıya doğru iki gözlem birimi biraraya gelip 1 küme oluşturur gibi ilerler yukarıya doğru birleştirilerek işlemler yapılır. Bölümleyici kümelemede ise tüm gözlemler bir arada olduğunda tek bir kümedir. aşşağa indikçe bölünerek her bir gözlem tek başına kalacak şekilde aşşağıya doğru yapılır.

#K Means ile farkı;  K-meansde dışarıdan müdahale edemiyoduk dolayısıyla gözlemleme şansımız yoktu ama Hiyerarşik kümelemede gözlemleme yapıp belirli noktalardan çizgiler çekerek görsel teknik üzerinden yeni kümeler yapabiliriz.

#veri setimizi okutuyoruz
df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)
#standartlaştırıyoruz
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

#linkage yöntemi ile agglomerative cluster yapıyoruz, euclide uzaklığında göre gözlem birimlerini kümelere ayırır
hc_average = linkage(df, "average")
#dendogram ile görselleştiriyoruz
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

#daha güzel bir görselleştirme içi, lastp ve truncate modunu aktif ettik. daha sade bir yapı oldu anlayılabilirliği kolaylaştı
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################
#y eksenindeki belirli bir alana çizgi attık ve kümelerimizi oraya göre belirliyebiliriz.

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering
#5 küme olarak karar verdim modeli kuruyorum
cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
#fit ediyorum ve 5 cluster için bilgiler alıyorum
clusters = cluster.fit_predict(df)

#veri setimi tekrardan okutup, cluster numaralarımı ekliyorum
df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters
#+1 ekliyorum
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df.head()

################################
# Principal Component Analysis
################################
#Temel fikri, çok değişkenli verinin ana özelliklerini daha az sayıda değişken ile temsil etmektir.
#Küçük miktarda bir bilgi kaybını göze alıp değişken boyutunu azaltmaktır.
#neden?
#Örneğin doğrusal regresyon probleminde çoklu doğrusal bağlantı probleminden kurtulmak istiyor olabiliriz
#örneğin yüz tanıma probleminde resimlere filtre yapma ihtiyacı hissediyor olabiliriz. yada buna benzer sebeplerle boyut indirgeme ihtiyacımız olduğunda kullanırız.
#ayriyetten çok boyutlu veriyi görselleştirmek içinde kullanılır.
#PCA veri setini bağımsız değişkenlerin doğrusal kombinasyonları ile ifade edilen bileşenlere indirger, dolayısıyla bu bileşenler arasında korelasyon yoktur.
#değişken gruplarının varyanslarını ifade eden öz değerler ve veri setindeki değişkenleri gruplandırır. en fazla varyansa sahip olan gruplar en önemli değişkenlerdir.

df = pd.read_csv("week7/datasets/hitters.csv")
df.head()
#sayısal değişkenler
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

#kategorik değişkenleri ve boş değerleri düşürdük
df = df[num_cols]
df.dropna(inplace=True)
df.shape

#standar scale ediyoruz df'imizi
df = StandardScaler().fit_transform(df)

#pc'yi uyguluyoruz
pca = PCA()
pca_fit = pca.fit_transform(df)

#bileşenlerin başarısına bakıyoruz gittikçe artıyor.
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# Optimum Bileşen Sayısı
################################
#görselleştirerek en optimum bileşen sayısına bakıyoruz
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()
#grafiğe bakınca 2-3 gibi bir değer tercih edilebilir. yoruma açık bir nokta

################################
# Final PCA'in Oluşturulması
################################
#3 sayısını seçtik ve final PCA'yi oluşturduk
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
#1. bileşen %0.97, 2 bileşen %0.01, 3. bileşen %0.004 'ünü açıklıyor
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# BONUS: Principal Component Regression
################################
#verimizi okutuyoruz ve shapelerine bakıyoruz
df = pd.read_csv("week7/datasets/hitters.csv")
df.shape
len(pca_fit)
#numeric cols'u yakalıyoruz
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)
#16 değişkeni 3 bileşene çevirmiştik, şimdi num colsun dışındaki değişkenleri yakalıyorum
others = [col for col in df.columns if col not in num_cols]
#3 bileşeni df'e çeviriyoruz
pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()
#önce bir PCA uygulanıyor ve değişkenlerin boyutu indirgeniyor, sonrasında bu bileşenlerin üzerine bir regresyon modeli kuruluyor.
df[others].head()
#2 df'i bir araya getiriyoruz
final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)
final_df.head()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#label encodeluyoruz çünkü kategorik değişkenlerimizin hepsi 2 sınıftan oluşuyor.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)
#NaN'ları atıyoruz
final_df.dropna(inplace=True)

#bağımlı ve bağımsız değişkenleri ayırıyoruz
y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

#LR regresyon modelimizi kruuyoruz direkt hataya bakıyoruz, fit etmiyoruz daha sonra kullanmıcağımız için
lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()

#DT regressor ilede deniyoruz aynı şeyi
cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV ile deniyoruz
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

df = pd.read_csv("week7/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("week7/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")
