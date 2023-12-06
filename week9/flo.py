######### Gözetimsiz Öğrenme ile Müşteri Segmentasyonu

#İş Problemi:
#Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering ) müşteriler kümelere ayrılıp davranışları gözlemlenmek istenmektedir.

#Veri Seti Hikayesi:
#Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden
#oluşmaktadır

#Değişkenler:
#master_id: Eşsiz müşteri numarası
#order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
#last_order_channel : En son alışverişin yapıldığı kanal
#first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date : Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
#store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage , fcluster
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)

######     Görev 1: Veriyi Hazırlama
#Adım 1: flo_data_20K.csv verisini okutunuz.

df=pd.read_csv("week9/datasets/flo_data_20k.csv")
df.head()

#Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
#Recency(yenilik) : müşterinin yenilik, yada bizden en son ne zaman alışveriş yaptığı durumunu ifade eder.
#Frequency(Sıklık) : müşterinin yaptığı alışveriş/işlem sayısı.
#Monetary(Parasal Değer):  Müşterilerin bize bıraktığı parasal değeri ifade eder


date_columns=df.columns[df.columns.str.contains("date")]
df[date_columns]=df[date_columns].apply(pd.to_datetime)
df.info()

analysis_date = df["last_order_date"].max() + pd.Timedelta(days = 2)

df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (analysis_date - df["first_order_date"]).dt.days
#df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')
df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["monetary"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]



model_df = df[["frequency","monetary","recency","tenure"]]
model_df.head()
# Modele girecek değişkenler ayrı bir dataframe'e aktarıldı

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(4, 1, 1)
check_skew(model_df,'frequency')
plt.subplot(4, 1, 2)
check_skew(model_df,'monetary')
plt.subplot(4, 1, 3)
check_skew(model_df,'recency')
plt.subplot(4, 1, 4)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.show(block=True)


######     Görev 2: K-Means ile Müşteri Segmentasyonu
#Adım 1: Değişkenleri standartlaştırınız.

sc = StandardScaler()
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()
#değerlerimizi 0 ile 1 arasında yerleştirdik


#Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans(random_state=42)
elbow = KElbowVisualizer(kmeans, k=(2, 10))
elbow.fit(model_df)
elbow.show(block=True)
# Siyah çizgi çekilen kısım bize en optimum küme sayısını belli ediyor.Bu grafiğe göre 5 olmalı.
elbow.elbow_value_

#Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
k_means = KMeans(n_clusters = 5, random_state= 42).fit(model_df)
segments=k_means.labels_
segments

final_df = df[["master_id","frequency","monetary","recency","tenure"]]
final_df["segment"] = segments + 1
final_df.head()
#Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"frequency":["mean","min","max"],
                                 "monetary":["mean","min","max"],
                                 "recency":["mean","min","max"],
                                 "tenure":["mean","min","max","count"]})

######     Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=20, color='r', linestyle='--')
plt.show(block=True)

# Burada y değerini 20 seçtiğimiz için 20 den çizgi çekti.Toplam 5 dal var.İsteğe göre daha üstten de çizgimizi
# çekebiliriz böylece dal sayısı daha fazla  olur

#Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id","frequency","monetary","recency","tenure"]]
final_df["segment"] = segments +1
final_df.head()
#Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"frequency":["mean","min","max"],
                                 "monetary":["mean","min","max"],
                                 "recency":["mean","min","max"],
                                 "tenure":["mean","min","max","count"]})
