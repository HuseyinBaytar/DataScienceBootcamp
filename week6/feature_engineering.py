#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################
#If your data is bad, your machine learning tools are useless
#the world's most valuable resource is no longer oil, but data.
#Applied machine learning is basically feature engineering.  -Andrew NG

#özellik mühendisliği: Özellikler üzerinde gerçekleştirilen çalışmalar. Ham veriden değişken üretmek.
#veri ön işleme: çalışmalar öncesi verinin uygun hale getirilmesidir.


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


dff = pd.read_csv("week6/datasets/application_train.csv")
dff.head()

df= pd.read_csv("week7/datasets/titanic.csv")
df.head()



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################
#verideki genel eğilimin oldukça dışına çıkan değerlere aykırı değer denir.
#aykırı değerler genel olarak, sektör bilgisine, standart sapma yaklaşımına, z skoru yaklaşımına ve boxplot/ LOF'a göre değerlendirilebilir.

#############################################
# Aykırı Değerleri Yakalama
#############################################
# Grafik Teknikle Aykırı Değerler
###################
#aykırı değer olduğunu görüyoruz.
sns.boxplot(x=df["Age"])
plt.show()
###################
# Aykırı Değerler Nasıl Yakalanır?
###################
#yaş değişkeninin 25'lik çeyreği
q1 = df["Age"].quantile(0.25)
#yaş değişkeninin 75lik çeyreği
q3 = df["Age"].quantile(0.75)
#25lik den 75'i çıkarıyoruz ve iqr değerini elde ediyoruz
iqr = q3 - q1
#up limiti için 1.5 ile iqr'ı çarpıp q3'le topluyoruz
up = q3 + 1.5 * iqr
#low limit için 1.5 ile çarpıp q1'den çıkarıyoruz
low = q1 - 1.5 * iqr

#yaş değişkeni low ve up sınırından küçük/büyük olanlar
df[(df["Age"] < low) | (df["Age"] > up)]
#indexlerini aldık
df[(df["Age"] < low) | (df["Age"] > up)].index

###################
# Aykırı Değer Var mı Yok mu?
###################
#herhangi bir aykırı değer var mı diye sorduk, True döndürdü
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
#küçükleri sorduk false dedi
df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################
#outlier yakalama fonksiyonu
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")
#fare değişkenindeki outlierları yukardaki fonksiyon ile yakalıyabiliyoruz
df2[(df["Fare"] < low) | (df["Fare"] > up)].head()
df2[(df["Fare"] < low) | (df["Fare"] > up)].index

#aykırı değer var mı? yok mu?'nun sorunu cevaplıcak bir fonksiyon
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff.head()
#columns ayırma fonksiyonu
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]             # kategorik değişkenleri yakala
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]                                                #numeric ama kategorik değişkenler
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]                                                #kategorik olup bir bilgi taşımayan yani çok fazla eşsiz olan
    cat_cols = cat_cols + num_but_cat                                                          #tüm kategorik değişkenleri topluyoruz
    cat_cols = [col for col in cat_cols if col not in cat_but_car]                             #kardinal olmayanları yakala

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]           # numerikleri yakala,
    num_cols = [col for col in num_cols if col not in num_but_cat]                          # numeric ama categoric olanları alma

    print(f"Observations: {dataframe.shape[0]}")                        # shape'in iyazdır
    print(f"Variables: {dataframe.shape[1]}")                             #variableların shape'ini
    print(f'cat_cols: {len(cat_cols)}')                                  #kategorikleri yazdır
    print(f'num_cols: {len(num_cols)}')                                 #numerikleri yazdır
    print(f'cat_but_car: {len(cat_but_car)}')                           #cardinalleri yazdır
    print(f'num_but_cat: {len(num_but_cat)}')                           #numeric ama kategorik olanları yazdır
    return cat_cols, num_cols, cat_but_car

#891 gözlem, 12 değişken, 6 kategorik, 3 numerik, 3 kategorik ama cardinal, numeric but cat 4 tane ama kategoriğin içindedir, o yüzden gözlem amaçlı printlenmiştir.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# passengerid bizim exceptionumuz, date değişkenide aynı şekilde olabilirdi, o yüzden passengerID'yi atıyoruz
num_cols = [col for col in num_cols if col not in "PassengerId"]

#outlier var mı diye soruyoruz, age ve fare'da var diyor
for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

#id değişkeni'ni atıyoruz
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, col_name, index=False):  #df ismini gircez, sütün ismini gircez, değişken ismini giricez ama öntanım false
    low, up = outlier_thresholds(dataframe, col_name)   #low ve up değerleri yakala

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:   #eğer 10'dan büyük threshold varsa
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())   #5 tanesini yazdır
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])          #eğer 10'dan küçükse direkt yazdır

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index         # index argümanı true ise  indexleri yazdır
        return outlier_index


#yaş değişkeninde 10dan fazla outlier olduğu için 5 tanesini döndürdü
grab_outliers(df, "Age")
#sadece indexleri döndürdü
grab_outliers(df, "Age", True)
#daha sonra kullanmak üzere saklamak istersek böyle saklıyabiliyoruz
age_index = grab_outliers(df, "Age", True)


#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################
# alt limit ve üst limitin üstünde olanların dışındakileri alıyoruz
low, up = outlier_thresholds(df, "Fare")
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

#alt sınırdan küçük ve üst sınırda nbüyük olanları atıp df'e eşitliyoruz
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

#columnsi yakaladık
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#passengerı attık
num_cols = [col for col in num_cols if col not in "PassengerId"]
#for döngüsü ile gezip, tüm outlierları attık.
for col in num_cols:
    new_df = remove_outlier(df, col)
df.shape[0]  -  new_df.shape[0]
#891 değer,  yeni dfde 775 değer

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################
#alt limitten aşşağıda olan değişkeni alt limitle, üst limitten yukarda olan değişkeni üst limitle değiştiriyoruz
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
#outlier var mı ? diye soruyoruz
for col in num_cols:
    print(col, check_outlier(df, col))
#outlierları değiştiriyoruz
for col in num_cols:
    replace_with_thresholds(df, col)
#tekrar soruyoruz
for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################
#diyelim ki verimizde, evlenme durumu var, 3 kere evlenmiş olması outlier mı? değildir
#ve yaş durumu var diyelim ki 17 yaşında, buda normaldir
#ama 17 yaşıdna 3 kere evlenmiş olması anormaldir

#local outlier factor= komşuluklara göre uzaklık hesaplama şansı sağlar,verilen gözlemler 1'den ne kadar uzaksa outlier ihtimali o kadar artar

df = sns.load_dataset('diamonds')  #diamonds veri seti
df = df.select_dtypes(include=['float64', 'int64'])    #sadece sayısallar
df = df.dropna()  #boşları at
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

#1889 outlire
low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape

#2545 outlier
low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

#tek kbaşına baktığımızda çok yüksek sayıda aykırılar geldi
#komşuluk sayısını 20 yapıyoruz
clf = LocalOutlierFactor(n_neighbors=20)
#fitliyip predictliyoruz
clf.fit_predict(df)

#scoreları tutuyoruz,
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

#elbow yöntemine göre belirliyebiliriz,
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

#eşik değeri 3 belirliyoruz
th = np.sort(df_scores)[3]

#eşik değerden küçük olanları aykırı değer olarak belirliyoruz
df[df_scores < th]
#3 tane geliyor
df[df_scores < th].shape

#3 tane gözlem var, neden aykırılığa takıldı?,
#1. gözlemimiz derinliği 78 olup, price'ı düşük olması problem göstermiş olabilir
#2.nin z'nin değeri 31,800 ama ortalaması 3.5
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

#index bilgilerini yakalayıp
df[df_scores < th].index
#silebiliriz istersek
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#baskılama yapılabilir ama baskılamak istersek, neyle baskılıcaz? bu seneryo için 3 gözlem var np ama 100'lerce outlier olursa baskılarsak veri setinin seyri değişebilir
#duplice kayıt yapılabilir. ama ciddi problemlere sebep olabilir.
## eğer ağaç yöntemleriyle çalışıyorsak, bunlara dokunmucaz, maybe outlier threshold fonksiyonlaryla çok ufak bir traş edicez
#azlık çokluk durumuna göre silinebilir ama yoruma dayalıdır.

#############################################
# Missing Values (Eksik Değerler)
#############################################
#gözlemlerde eksiklik olması durumunu ifade etmektedir.
#eksik değerler nasıl çözülür?  -silme   -Değer atama    -Tahmine dayalı yöntem

#############################################
# Eksik Değerlerin Yakalanması
#############################################
df.head()
# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()
# degiskenlerdeki eksik deger sayisi
df.isnull().sum()
# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()
# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]
# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]
# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)
#boş olan frekansların oranını hesaplamak
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
#içinde boş olan gözlemleri na_cols'a atayıp, na_cols'u yazdırdığımız zaman, boş değişken olan featureların isimlerini görürüz
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False): #eğer true dersek bize eksik değişkenlerin isimlerini verir
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]   # eksik değişkenlerin olduğu colunmsu yakala
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)   #eksik değer sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)      #eksik değer oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])       # birleştir
    print(missing_df, end="\n")  #df'i yazdır
    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)

#############################################
# Eksik Değer Problemini Çözme
#############################################
###################
# Çözüm 1: Hızlıca silmek
###################
#boş olan değerleri düşürmek için dropna() yaptıktan sonra atama yaparız
#ama dropna yaparken eğer bir gözlem biriminde herhangi bir yerinde nan varsa tüm gözlemi düşürürüz o yüzden çok veri kaybedilebilir
df.dropna().shape
###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################
#fillna ile, boşları ortalamasına, medyanına veya 0 ile doldurabiliriz
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

#eğer elimizde bir çok boş değişkenli bir veri seti varsa
#object tipler olduğu için hata vericektir
df.apply(lambda x: x.fillna(x.mean()), axis=0)
#burdaki sadece sayısal değişkenleri doldurmak için objectden farklı olanları doldur diyoruz
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
#atama yapıyoruz
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
#büyükten küçüğe sıralıyoruz
dff.isnull().sum().sort_values(ascending=False)
#kategorik değişkenler için en mantıklı doldurma, mod'unu almaktır
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
#veya özel ifade olarak 'missing' ile doldurabiliyoruz
df["Embarked"].fillna("missing")
# koşulumuz, eğer kategorik değişken ve aynı zamanda eşssiz değer sayısı 10'dan küçük ise, mode2u ile doldur değilse(else) kendi halinde bırak diyoruz
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
###################
# Kategorik Değişken Kırılımında Değer Atama
###################
#veri setindeki bazı değişkenleri kırılım olarak ele alıp, ona göre atamalar yapıyoruz, mesela female'in yaş ortalaması 27, erkeklerin 30
df.groupby("Sex")["Age"].mean()
#yaşın ortalaması 29
df["Age"].mean()
#bütün eksiklere 29 değerini atamak yerine, kadınlarda eksiklik olunca onlara 27, erkeklerin eksik olanlarına 30 atayabilriz

#kırılımla gelen cinsiyetle ilgili ortalamaları, fillna ile age değişkenindeki boşluklara dolduruyoruz, yani kadınların ortalaması kadınlara, erkeklerin ortalaması erkeklere
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))

#buda fillna'sız uzun bir şekilde yapmak, kadınların ortalamasını kadınlara
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
#erkeklerin ortalamasını erkeklere dolduruyoruz
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################
#columnsları yakalıyoruz
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#numeric olarak passenger id'yi almıştı onu atıyoruz
num_cols = [col for col in num_cols if col not in "PassengerId"]
#get dummies methodu ile değişkenleri ayırırız
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
#iki sınıflı yada daha fazla sınıflı değişkenleri numeric bir şekilde yazdırmaya çalışıyoruz
dff.head()

# değişkenlerin standartlatırılması
#değerleri 0-1 arasına getir diyoruz
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
#knn ile eksik olan değerlerin doldurulması tahmien dayalı
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
#standartlaştırdık ama minmaxladığımız için 0-1 arasında oluyor
#bu bölümde minmaxi geri alıp
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

#ilk df'i, 2. dife atıyoruz
df["age_imputed_knn"] = dff[["Age"]]
#nereye ne atadık ona bakıyoruz
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#############################################
# Gelişmiş Analizler
#############################################
###################
# Eksik Veri Yapısının İncelenmesi
###################
#missing no kütüphanesiyle eksik verileri görebiliyoruz
msno.bar(df)
plt.show()
#matrix methoduylada bakabiliyoruz, eğer değişkenliklerdeki eksiklikler bir aralıkta geliyorsa,bunu gözlemliyebiliyoruz
msno.matrix(df)
plt.show()
#ısı haritası ile eksiklikler üzerine kuruludur, eksik değerler belirli bir korelasyon ile ortaya çıkmış mıdır?
msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################
#veri setimdeki eksikliklerin, bağımlı değişken tarafında bir karşılığı var mı,
missing_values_table(df, True)
na_cols = missing_values_table(df, True)
#eksik değeri  olan değişkenleri çektik

def missing_vs_target(dataframe, target, na_columns):                  #df, target ve boş kolumnslar
    temp_df = dataframe.copy()                                              #geçici bir df açıp df'in kopyası veriyoruz
    for col in na_columns:                                                            #na'e sahip olan noktaları FLAG'le diyoruz
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)       #eksik değere sahip olan noktalara 1, boş olanlara 0 yazdık
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns           #temp df'de bir seçip yap, tüm sütunları seç ama içinde NA ifadesi olanları getir diyoruz
    for col in na_flags:                                                                       #bu değişkenlerde gez
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")      #targetin meanını al ve değişkenin meanini,countunu al

missing_vs_target(df, "Survived", na_cols)


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################
#encode etmek: değişkenlerin temsil şekillerinin değiştirilmesi
#############################################
# Label Encoding & Binary Encoding
#############################################
#örneğin önceden male/female olan bir değişkeni  0/1 yapmak
#veya diyelim ki eğitim değişkeni var ordinal, sıralı şekilde 0 dan 5'e label encoder yapıyoruz bunu yapmak için değişkeni sayısal olarak elimizle atıyabiliriz.
#one hot encoder ilede yapılabilir.

#label encoderı tanımlıyoruz
le = LabelEncoder()
#label encoderı fit edip transformluyoruz
le.fit_transform(df["Sex"])[0:5]
#burdada ger içeviriyoruz
le.inverse_transform([0, 1])

#hızlı bir şekilde fonksiyon ile yapabiliyoruz
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

#unique yaptığımızda eksik değeri getirir, nunique eksik değeri sınıf olarak görmez
#binary yani sadece 2 farklı değişkeni olan değerleri yakalıyoruz
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]and df[col].nunique() == 2]
#for döngüsü ile hepsini label encoderdan fonksiyon ile geçiriyoruz
for col in binary_cols:
    label_encoder(df, col)

#eğer eksik değer varsa, label encodeda 3 olur mesela, female 0, male 1, NAN 2 olur.

#3 sınıfı var ama len aldığımızda 4 sınıf gözükür
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

#############################################
# One-Hot Encoding
#############################################
#sınıflar arası fark olmayan değişkenlerde, yani mesela takımlar (real madrid,barca,bjk) vs bunlara 0-1-2 dersek, 2 en yüksek olucak, o yüzden tüm sınıfları değişkenlere dönüştürürüz.

df["Embarked"].value_counts() #3 değişkeni var
#drop first diyerek dummy değişken tuzağından kurtuluruz yani ilk sınıf drop edilir.
#yani real madridi droplar, barca ve bjk kalır eğer veri setinde barca ve bjk'ye puan yoksa model anlıyarak o puanı real madride yollar.

#get_dummies methodu ile değişkeni dummylere dönüştürür
pd.get_dummies(df, columns=["Embarked"]).head()
#drop first ile ilk gözlemi atarız, dummy değişken tuzağına düşmemek için
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
#eğer ilgili değişkendeki eksik değişkenlerde bir sınıf olarak gelsin dersek, eksik değerler içinde sınıf oluşturur, dummy_na True dersek
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
#cinsiyet değişkeni 2 değişkenliydi ama burda sadece male gelir, erkek mi ?(1 ve 0), ve aynı şekilde yandan embarked'ı ekledik onuda encodeladı
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#unique değeri 2'den yüksek ve 10'dan küçük olanları yakala ohe cols'a ata diyoruz
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
#fonksiyonu ohe_cols'a yapıyoruz
one_hot_encoder(df, ohe_cols).head()

#############################################
# Rare Encoding
#############################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.
###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################
#yukardaki büyük veri seti olarak 2. veri setini çalıştırıyoruz
#columnsları yakalıyoruz
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):                                                   #kategorik değişkenlerin yakalanması
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))       #kategorik değişkenlerin sınıflarını ve oranlarını getircek bir fonksiyon
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)                                         #görselleştirmek için
        plt.show()

#for döngüsüyle kategoriklerin değişkenleri ve oranları gelir.
for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################
#rare kategoriler vardı bu değişkende 5-10 tane valuesi olan
df["NAME_INCOME_TYPE"].value_counts()
#bu rare'ların target ile arasındaki ilişkiye bakıyoruz
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

#tüm kategorikler için rare analizi yapabiliceğmiz bir fonksiyon
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()                                                                           #geçici df'e atıyoruz
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]         #eğer fonksiyona girilen rare oranından daha düşük sayıda herhangi bir sınıf varsa,rare kolonlar olarak getir diyoruz
    for var in rare_columns:                                                                             # rareların içinde gez
        tmp = temp_df[var].value_counts() / len(temp_df)                                                 # sınıf oranları hesaplandı
        rare_labels = tmp[tmp < rare_perc].index                                                         # rare kolonlardaki oranlardan daha düşük olanları yakala ve, indexleri al
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])                 #gördüğün rare'ları al ve hepsinin yerine Rare yaz
    return temp_df
#new_df'e atıyoruz rare'lı olanları
new_df = rare_encoder(df, 0.01)
#targeta göre olanlara bakıyoruz
rare_analyser(new_df, "TARGET", cat_cols)

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################
#değişkenler arasındaki ölçüm farklılığını gidermek, eşit şartlar altında yaklaşsın diye
###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################
#titanic veri seti
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

#veri setindeki içinde yaş değişkenlerini yakalıyoruz
age_cols = [col for col in df.columns if "Age" in col]
#num summary fonksiyonumuz ile grafik alıyoruz
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning
###################
df["Age_qcut"] = pd.qcut(df['Age'], 5)
#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################
#ham veriden değişken üretmek
#örnek mesela timestamp değişkeninden, yıl ,ay,gün vs gibi yeni değişkenler çıkarılabilir
#############################################
# Binary Features: Flag, Bool, True-False
#############################################
#titanici importluyoruz
#na'lere 0, diğer değişkenlere 1 yazıyoruz
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
#yeni oluşturduğumuz değişkene göre, survivedin meanini alıp bakıyoruz ve, cabin numarası olmayanların yaşama oranı daha yüksek gözüküyor.
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

#yeni oluşturduğumuz feature'ın arasındaki oranın dağılımları anlamlı bir dağılım mıdır? test ediyoruz
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p value 0 döndürdü, h0 hipotezi red edildi yani aralarında anlamlı bir farklılık vardır.

#sibling ve parentları toplayıip, yeni bir değişken oluşturuyoruz , alone mu değil mi binaryliyoruz
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
#survived'e göre oranına bakıyoruz
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

#istatistiki olarak anlamlı bir fark varm ı?
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#h0 red edilir yani anlamlı bir fark vardır.

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################
###################
# Letter Count
###################
#bir isimde kaç tane isim olduğunu saydırıyoruz
df["NEW_NAME_COUNT"] = df["Name"].str.len()
###################
# Word Count
###################
#bir isimde kaç kelime var onu sayıyoruz / stringlere çevir, split ile boşluklara ayır, sonrasında len ile sayısını al
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
###################
# Özel Yapıları Yakalamak
###################
#doktor ismine sahip olanları yakalamaya çalışıyoruz. / split et ve her gezdiğin isimde Dr geçiyosa onu al
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
#doktora göre groupby alıp survived'ın ortalamasına bakıyoruz, doktor olanların hayatta kalma oranı yüksek ama 10 tane doktor var.
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################
#title'lar bizim için önemli olabilir o yüzden, regex ile o titleları çekiyoruz, paternini verip boşluk nokta küçük büyük harfler vs
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#alttaki 3'lüye göre new title groupby al ve surviveda göre mean,age'e göre count/mean alıyoruz ve inceliyoruz.
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()
#değişkeni datetime'e çeviriyoruz
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
# year
dff['year'] = dff['Timestamp'].dt.year
# month
dff['month'] = dff['Timestamp'].dt.month
# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year
# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month
# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
#titanic
#yaş ile pclassı çarpıyoruz, refah açısından yaşı küçük ama yüksek sınıfta olan birşey ifade ediyor olacaktır.
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
#akrabalık sayıları + kişinin kendisi 1  = gemideki aile bireylerinin sayıları
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
#yaşı 21'den küçük olan erkekler
df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
#olgun erkekler
df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
#yaşlı erkekler
df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
#genç kızlar
df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
#genç kadınlar
df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
#olgun kadınlar
df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'
#yaşlı kadınlar
#survived'a göre ortalamasını alıp baktığımızda, olgun kadınların hayatta kalma olasılığı yüksek
df.groupby("NEW_SEX_CAT")["Survived"].mean()

#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################
#titanic bütün columnları büyütoyurz
df.columns = [col.upper() for col in df.columns]
#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################
# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

#değişkenleri ayırdığımız fonksiyonu yazıyoruz
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#passengerid'yı atıyoruz
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
#############################################
# 2. Outliers (Aykırı Değerler)
#############################################
#yukardaki fonksiyonlarla outliersi atıyoruz
for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outlier(df, col))
#############################################
# 3. Missing Values (Eksik Değerler)
#############################################
#fonksiyon ile eksik değerleri sorduruyoruz
missing_values_table(df)
#cabini dropluyoruz, yeni değişkenimiz olduğu için
df.drop("CABIN", inplace=True, axis=1)
#bunlarıda düşüyoruz
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

#new title'a göre yaş değişkeninin eksik değerlerini dolduruyoruz
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

#tipi object olan ve eşsiz değer sayısı 10'dan olanları dolduruyoruz
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################
#2den küçük olan kategorikleri yakalıyoruz
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
#ve fonksiyonumuza verip label encodeluyoruz
for col in binary_cols:
    df = label_encoder(df, col)
#############################################
# 5. Rare Encoding
#############################################
#analyser fonksiyonumuzla bakıyoruz
rare_analyser(df, "SURVIVED", cat_cols)
#0.01'e göre birleştiriyoruz
df = rare_encoder(df, 0.01)

#############################################
# 6. One-Hot Encoding
#############################################
#10'dan küçük,  2'den büyük olanları yakalıyoruz
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
#fonksiyonumuzu verip dönüştürüyoruz
df = one_hot_encoder(df, ohe_cols)
#oluşturudğmuz yeni değişkenler var ama tekrardan rare analyzser ile bakacaz gerekli mi gereksiz değişkenler mi ona bakıcaz
#tekrardan kategorikleri numerikleri vs yakalıyoruz
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#passender id'yi atıyoruz
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
#fonksiyonumua sokuyoruz
rare_analyser(df, "SURVIVED", cat_cols)
#işimize yaramayan columnları yakalıyoruz mesela, aile sayısı 11 mi olan değişekn bir  işimize yaramıyor
#oranı 0.01'den düşük olanları yakalayıp atıyoruz
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################
#bu problemde gerekli değil ama ihtiyacımız olursa, bu şekilde standartlaştırabiliriz
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()
#############################################
# 8. Model
#############################################
#veri setini ikiye ayırıyoruz
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
#randomforest modelini kurup tahmin ettiriyoruz
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#%80 accuracy ulaştık
#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################
dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#%70 veriyor

# Yeni ürettiğimiz değişkenler ne alemde?
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


