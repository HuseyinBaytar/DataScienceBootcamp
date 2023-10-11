############ PANDAS SERIES ####################
import pandas as pd
#pandas serisi tek boyutlu ve index bilgisi barındıran veri tipi
#pandas dataframe'i iki boyutlu ve index bilgisi barındıdan veri tipi
s = pd.Series([10, 77, 12, 4, 5 ])
type(s)
s.index   #index bilgilerini görürüz
s.dtype   #tipini görürüz "int64"
s.size    #toplam eleman sayısını görürüz
s.ndim    #kaç boyutlu olduğunu görürüz
s.values  #içindeki verilere erişiriz
type(s.values)  #numpy array olarak döndürür.
s.head(3)  #3 verdiğimz için ilk 3'ü görürüz
s.tail(3)  # 3 verdiğimiz için son 3'ü görürüz

########### Veri okuma #####################
# pd.read_csv("buraya dosyanın path'ini yazmamız gerekiyor")
# pd.read_excel
# pd.read_json

############ Veriye hızlı bakış  #####################
import seaborn as sns
df = sns.load_dataset("titanic")
df.head() #hızlı bakış için ilk 5 indexi alıp bakarız
df.tail() #sondaki 5 değer
df.shape #satır sütün bilgisi
df.info()  #değişkenler, değişken tipleri, ve boş olup olmadığı
df.columns  #değişkenlerin isimleri
df.index   #index bilgisi
df.describe().T  #sayısal bilgilerin özet bilgileri
df.isnull().values.any() #eksik değer var mı diye sorduk
df.isnull().sum()   #her değişkenin içindeki boş değerleri sorarız
df["sex"].value_counts()  #cinsiyet değişkeninde kaç erkek kaç kadın var ona baktık

############# Seçim işlemleri  ####################
df[0:13]  #0 dan 13'e kadar gider
df.drop(0, axis=0).head() #0. değeri attık kalıcı değil ama, kalıcı olması için inplace true yazmak lazım
# değişkeni indexe çevirmek = örneğin yaşı indexe atamak
df.index = df["age"]
df.head()
df.drop("age", axis=1, inplace=True)
#indexi değişkene çevirmek
# df.["age"] = df.index  #index yaptığımız yaşı, tekrardan dataframe'e soktuk
df.reset_index() # indexdeki age değişkenini silmenin öbür yolu

# değişkenler üzerinde işlemler
"age" in df   #bu değişken bu veri setinin içinde var mı ?
df["age"].head() #aynı şekilde veri setinin içinde olup olmadığını görürüz
#tek değişken seçerken döndürdüğü type pandas seriesdir. fonksiyonlara data frame için yazıyorsak eğer çift parantez kullanılmalı
type(df["age"])
type(df[["age"]])

df[["age","alive"]]  # birden fazla değişken
#listeye atayıp df[liste] ile de bakılabilir
#yeni değişken atamak
df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]
#değişken silmek  // aynı şekilde liste verilerek silebiliriz
df.drop("age3",axis=1)

df.loc[:, df.columns.str.contains("age")]                       # içerisinde yaş değişkeni bildirenleri seçti
df.loc[:, ~df.columns.str.contains("age")]                      #içerisinde yaş değişkeni barındırmayan hepsini seçti

#iloc / loc
#iloc = integer based selection   # indexi seçer
df.iloc[0:3]   #0'dan 3'e kadar
df.iloc[0,1]

#loc: label based selection       #columnları seçer
df.loc[0:3]

df.loc[0:3, "age"] #yaş değişkeninde 0dan 3'e gezdik

#liste atayıp aynı şekilde loc için bakarken o listeyi verip bakabiliriz
col_names = ["age","embarked","alive"]
df.loc[0:3, col_names]

#koşullu Seçim
#öğreniğin veri setinde yaşı 50 den büyük olanları seçmek istiyoruz
df[df["age"]>50].head()  #yaşı 50den büyük olan kişiler
df[df["age"]>50]["age"].count()  #yaşı 50den büyük olan kişi sayısı

df.loc[df["age"]>50, ["age","class"]].head()   #classlarına göre, yaşı 50'den büyük olanlar ve yaşları
#yaşı 50den büyük olan ve cinsiyeti erkek olanlar
df.loc[(df["age"]>50) & (df["sex"] == "male"), ["age","class"]].head()
#birden fazla koşul giriliyosa koşullar parantez içine alınmalıdır ***********


############ Toplulaştırma ve gruplama #####################
df["age"].mean()  #yaş ortalaması gelir
df.groupby("sex")["age"].mean()   #cinsiyet kırılımında, yaş değişkeninin ortalamasını aldık
df.groupby("sex").agg({"age" : ["mean","sum"]})  # cinsiyet kırılımında, age'in  ortalama ve toplamını aldık, daha fazla fonksiyon kullanilabilr
df.groupby("sex").agg({"age" : ["mean","sum"],
                       "survived": "mean"})  # cinsiyet kırılımında, age'in mean,sum alıp birde, survived'ın meanini aldık

df.groupby(["sex","embark_town"]).agg({"age" : "mean",
                       "survived": "mean"})           # 2 kırılımlı bir groupby işlemi yaptık

# pivot tablo
df.pivot_table("survived", "sex",  "embarked")  # hücrelerin kesişimi olarak ortalamasını alır default değer
df.pivot_table("survived", "sex",  "embarked", aggfunc="std")  #standart sapmasını aldık

df.pivot_table("survived", "sex",  ["embarked","class"])
#0-10, 11-18, 19-25, 26-40, 41-90 olmak üzere yaşları böldük ve yeni feature'a atadık
df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90])
# cut = verilen değerlere göre atar,  qcut = %'lik değerlere göre atar
df.pivot_table("survived","sex","new_age")


############ Apply ve Lambda  #####################

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5
#lambda ile age değişkeni olanları 10'a bölüp yazdırdık
df[["age", "age2", "age3"]].apply(lambda  x: x/10).head()
#apply fonksiyonu ile fonksiyon yazmadan hizlica yazdırdık
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()


############ Birleştirme işlemleri  #####################
import numpy as np
m = np.random.randint(1,30 , size=(5,3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])
df2 = df1+99
#df1 ile df2'yi birbirine birleştiriyoruz
pd.concat([df1,df2])
#ignore index ilede iki listenin farkli indexlerini kaldirip tek index yapiyoruz
pd.concat([df1,df2], ignore_index=True)


df1 = pd.DataFrame({"employees": ["john", "dennis","mark","maria"],
                    "group":  ["accounting","engineering","engineering","hr"]})

df2 = pd.DataFrame({"employees": ["john", "dennis","mark","maria"],
                    "start_date":  [2010,2009,2014,2019]})

#emplooyesi baz alarak 2 grubu birleştirir
pd.merge(df1,df2)





