
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)   #bütün sütunları gör
pd.set_option('display.max_rows', 500)     #bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.4f' % x)    #virgülden sonra kaç tane sayı görmek istiyorsak
pd.options.display.width = 1000

df = pd.read_csv("week3/flo_data_20k.csv")

           # 2. Veri setinde
# a. İlk 10 gözlem,
df.head(10)

# b. Değişken isimleri,
df.columns

# c. Betimsel istatistik,
df.describe([0.01, 0.99]).T

# d. Boş değer,
df.isnull().sum()


# e. Değişken tipleri, incelemesi yapınız.
df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df['frequency'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]     #toplam işlem sayısı = frequency

df['monetary'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]   #ödenilen toplam  para = monetary


# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
df.head()
#time_list = df.columns[df.columns.str.contains("date")]
timelist = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]

df[timelist] = df[timelist].apply(pd.to_datetime)


# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.


df.groupby("order_channel").agg({"frequency":["sum","mean"],
                                 "monetary":["sum","mean"]})


# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"monetary": "max"}).sort_values("monetary", ascending=False).head(10)


# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"frequency": "max"}).sort_values("frequency", ascending=False).head(10)


# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def onhazirlik(df):
    df['frequency'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['monetary'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    time_list = df.columns[df.columns.str.contains("date")]
    df[timelist] = df[timelist].apply(pd.to_datetime)
    return df

# GÖREV 2: RFM Metriklerinin Hesaplanması

df["last_order_date"].max()
# today_date = dt.datetime(2021, 5, 29)

today_date = df["last_order_date"].max() + pd.Timedelta(days = 2)

#Recencyi bulmak için today date'den groupbya aldıktan sonra her bir müşterinin max tarihini bulmamız lazım ve today date'den çıkardınca Recency buluruz.
#Frequency için customerID'ye göre groupbya aldıktan sonra her bir müşterinin eşsiz fatura sayısına gidersek, müşterinin kaç tane satın alma yaptığını buluruz
#Monetary için customerID'ye göre groupby yaptıktan sonra totalprice'ların sum'unu alırsak, her bir müşterinin kaç para bıraktığını buluruz

rfm = df.groupby("master_id").agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                     'frequency' : lambda num: num,  #unique olduğu için direkt kendisini veriyoruz
                                     'monetary': lambda TotalPrice: TotalPrice}) #unique olduğu için direkt kendisini veriyoruz


rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()
rfm.describe().T



# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm["recency_score"] = pd.qcut(rfm['Recency'],5, labels=[5,4,3,2,1])

rfm["frequency_score"]= pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'],5,labels= [1,2,3,4,5])


rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T
rfm.head()


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalist',
    r'5[4-5]': 'champions',
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


rfm.head()



# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment","Recency","Frequency","Monetary"]]. groupby("segment").agg("mean")


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
# ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
# yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.


yeni_marka_hedef_musteri_id = pd.merge(rfm[(rfm["segment"] == "loyal_customers") |(rfm["segment"] == "champions")],df["master_id"][df["interested_in_categories_12"].str.contains("KADIN")],how='inner', on="master_id")

new_df = pd.DataFrame()
new_df["new_customer_id"] = yeni_marka_hedef_musteri_id["master_id"]

new_df.to_csv("yeni_marka_hedef_musteri_id.csv",index=False)


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

indirim_hedef_musteri_ids = pd.DataFrame()

dff = pd.merge(rfm[(rfm["segment"] == "new_customers") |(rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "cant_loose")],
         df["master_id"][df["interested_in_categories_12"].str.contains('ERKEK') & df["interested_in_categories_12"].str.contains('COCUK')],
        how='inner', on="master_id")

indirim_hedef_musteri_ids["new_customer_id"] = dff["master_id"]

new_df.to_csv("indirim_hedef_musteri_ids.csv")


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def create_rfm(df):
    df['frequency'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['monetary'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    time_list = df.columns[df.columns.str.contains("date")]
    df[timelist] = df[timelist].apply(pd.to_datetime)
    today_date = df["last_order_date"].max() + pd.Timedelta(days=2)
    rfm = df.groupby("master_id").agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                       'frequency': lambda num: num,
                                       'monetary': lambda TotalPrice: TotalPrice})
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalist',
        r'5[4-5]': 'champions',}
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    return rfm

