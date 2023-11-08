# Olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlama

# Ortalama (Average)
# Time-Based Weighted Average
# User-Based Weighted Average
# Weighted Raiting


#uygulama: Kullanıcı ve zaman ağırlıklı Kurs Puanı hesaplama

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import  MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)


#veri bilimi ve machine learning kursu veri seti
#puan: 4.8(4.764925)
#toplam puan: 4611
#puan yüzdeleri: 75,20,4,1,<1
#Yaklaşık Sayısal Karşılıkları: 3458, 922, 184 , 46 ,6

df = pd.read_csv("week4/course_reviews.csv")
df.head()
df.shape

#raiting dağılımı / hangi puandan kaçar tane
df["Rating"].value_counts()

#soru sorulan insanların dağılımı / kaçar tane soru sorunmuş
df["Questions Asked"].value_counts()

#soruların sorulara göre verdikleri puanın ortalaması

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating":"mean"})


#ortalam puan
df["Rating"].mean()

#böyle direkt ortalama aldığımız zaman, bu durumda ilgili ürünler ile ilgili müşteriler açısından son zamanlardaki memnuniyet trendini kaçırıyor oluruz.
#örneğin ürün ortaya çıktığı ilk zamanlarda çok yüksek puan alıp, ileride daha düşük puan alması, ağırlığını ilk zamanlar koruyacağı için, son zamanlardaki düşük puan aldığı trend kaçıyor olacaktır


######################################
#Time- Based Weighted Average
#puan zamanlarına göre ağırlıklı ortalama yaparsak eğer, zamana göre yüksek veya düşük aldığı puan trendlerini kaçırmayız
#######################################
#zaman değişkenini  datetime dtype'ına çevirdik
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
#şuanın tarihi olarak  veri setindeki son günün tarihini verdik
current_date = pd.to_datetime('2021-02-10 0:0:0')
# günümüzün tarihinden - veri setindeki tarihleri tek tek çıkarıp, days atlı yeni oluşturuğumuz feature'a kaydettik
df["days"] = (current_date - df["Timestamp"]).dt.days
#30 gün ve 30 günün aşşağısındaki sayıları çağırdık
df[df["days"] <= 30].count()

#30 gün ve 30 günün aşşağısındaki sayıları çağırdık ve ortalamasını almak için
df.loc[df["days"] <= 30,"Rating"].mean()

#30 dan büyük ve 90'a eşit/küçük olanların ortalmasını alalım
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

#90dan büyük 180'den küçük/eşit
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

#180 gün üstü tüm oylamalar
df.loc[df["days"] > 180,"Rating"].mean()

#0-30 gün = 4.77,
#30-90 gün = 4.76,
#90-180 gün = 4.75
#gördüğümüz üzere son zamanlarda kursun memnuniyeti ile ilgini artış var.

#30 gün altını %28,  30-90 arasını %26,  90-180 arasını %24 ve geri kalanıda %'nin kalanı olarak 22'ye bölüp döndürdğümüzde;

df.loc[df["days"] <= 30,"Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[df["days"] > 180,"Rating"].mean() * 22/100

#zaman aralıklı ölçüm yapıp 4.76 'ortalamayı alırız


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24,w4=22):
    return dataframe.loc[dataframe["days"] <= 30,"Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180,"Rating"].mean() * w4 / 100

time_based_weighted_average(df)

##################################################
#User-Based(user-quality/ user-rank) Weighted Average
#kullanıcı temelli ağırlıklı ortalama
#################################################
#acaba tüm kullanıcıların verdiği puanlar aynı ağırlıa mı sahip olmalı?
#örneğin kursun tamamını izliyen kişi ile kursun %1'ini izliyen kişi aynı ağırlığa mı sahip olmalı?

#farklı izleme oranlarında farklı puanlar var
df.groupby("Progress").agg({"Rating": "mean"})
#ilerleme durumu ile verilen puanların artışı ver gibi gözüküyor.

df.loc[df["Progress"] <= 10,"Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/ 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[df["Progress"] > 75,"Rating"].mean() * 28 / 100

#progresi 10'dan küçük olanlara %22 ağırlık, 10-45 arasına %24,  45-75 arasına %26 ve 75 üzerine %28 ağırlık verdim
# 4.80 ortalama döndürdü , kursun sonuna kadar izliyenlerin verdiği puan ortalamayı yüksek verdiği için.
#kursu tamamen izliyen bir kişi kursu iyi tanıyodur, bu kişinin verdiği puan ile, düşük sırada izliyen kişinin yorumu arasında fark vardır.



def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return  dataframe.loc[dataframe["Progress"] <= 10,"Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[dataframe["Progress"] > 75,"Rating"].mean() * w4 / 100


user_based_weighted_average(df)


#normalde direkt ortalamaya bakıyoken, muhattaplarımıza diyeceğimiz şeyler var örneğin,ürünlerin beğenilme trendlerini zamana göre hassaslaştırdık bununla beraber user quality score'umuz olsun dedik ve birden fazla faktöre göre  kullanıcılara özel skorlar oluşturup ağırlıklar verdik.

#kalite metriğimiz yani progress veya days metriği, nasıl inceleme yapıcağımıza göre değişir, sektörden sektöre değişir,  days = zaman ağırlıklı inceleme / progress = kullanıcı ağırlıklı inceleme olarak ele aldık bu konuda



##########################################
# Weighted Rating
##########################################
#tek fonksiyonda ikisinide hesaplamak;
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100  + user_based_weighted_average(dataframe)* user_w/100


course_weighted_rating(df)

#eğer bana göre user quaility daha önemli dersek
course_weighted_rating(df, time_w=40, user_w=60)
