##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
import pandas as pd
pd.set_option('display.max_columns', None)   #bütün sütunları gör
pd.set_option('display.max_rows', 500)     #bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.4f' % x)    #virgülden sonra kaç tane sayı görmek istiyorsak
pd.options.display.width = 1000

df_ = pd.read_csv("week3/flo_data_20k.csv")
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 +1.5 * interquantile_range
    low_limit= quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable]  = up_limit



# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

df.describe([0.01, 0.99]).T

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")




# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df['frequency'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df['monetary'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()

time_list= ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]

df[time_list] = df[time_list].apply(pd.to_datetime)



# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
import datetime as dt

df["last_order_date"].max()

today_date = df["last_order_date"].max() + pd.Timedelta(days = 2)



# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
df["recency"] =  (df["last_order_date"] - df["first_order_date"]).dt.days
df["T"] =  (today_date - df["first_order_date"]).dt.days



cltv_df = df.groupby('master_id').agg(recency_cltv_weekly = ("recency", lambda x: x // 7),
                                      T_weekly= ('T', lambda x: x // 7),
                                      frequency=('frequency', 'sum'),
                                      monetary_cltv_avg=('monetary', 'mean'))

cltv_df["monetary_cltv_avg"] = round(cltv_df["monetary_cltv_avg"] / cltv_df["frequency"])



# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


cltv_df.shape

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"]= bgf.predict(4*3,
                                            cltv_df['frequency'],
                                            cltv_df['recency_cltv_weekly'],
                                            cltv_df['T_weekly'])
cltv_df.head()

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"]= bgf.predict(4*6,
                                            cltv_df['frequency'],
                                            cltv_df['recency_cltv_weekly'],
                                            cltv_df['T_weekly'])

cltv_df.head()

# 3.ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

cltv_df.sort_values(by="exp_sales_3_month", ascending=False).head(10)

cltv_df.sort_values(by="exp_sales_6_month", ascending=False).head(10)



# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg'])




# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                      cltv_df['frequency'],
                                      cltv_df['recency_cltv_weekly'],
                                      cltv_df['T_weekly'],
                                      cltv_df['monetary_cltv_avg'],
                                      time= 6, #aylık
                                      freq='W', #T'nin frekans bilgisi
                                      discount_rate=0.01 )

cltv_df.head()

# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values(by='cltv',ascending=False)["cltv"].head(20)


# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması

# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.

cltv = cltv.reset_index()
cltv_df["segment"] = pd.qcut(cltv_df["cltv"],4, labels=["D","C","B","A"])

# 3. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_df.groupby("segment").agg({"recency_cltv_weekly":"mean","frequency":"mean", "monetary_cltv_avg":"mean"})



# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

cltv_df.groupby("segment").agg({"cltv" : ["count","mean","sum"]})

# D ve A segmentine bakınca  birinin çok yüksek ve birinin çok düşük olduğunu görüyoruz, fakat C ve B segmentlerine verilecek ilgi ve yapılan kampanyalar ile,
# alışverişe daha çok teşfik edilebilir. A segmentinin sum ve meanine bakılınca, diğer 3 segmentten çok daha fazla yatırım yaptıklarını görüyoruz, şirketten oldukça memnunlar
#onları dahada teşvik etmek için özel bir memberclub yada o segmentin müşterilerine özel indirimler yapılabilir.




# BONUS: Tüm süreci fonksiyonlaştırınız.

def create_cltv_p(dataframe,month=3):
    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")
    df['total_order'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['total_purch'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df['total_order'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['total_purch'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df['first_order_date'] = pd.to_datetime(df['first_order_date'])
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
    df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])
    today_date = dt.datetime(2021, 6, 1)
    df["lastfirst"] = (df["last_order_date"] - df["first_order_date"]).days
    df["firstandtoday"] = (today_date - df["first_order_date"]).days
    cltv_df = df.groupby('master_id').agg(recency_cltv_weekly=("lastfirst", lambda x: x.dt.days // 7),
                                          T_weekly=('firstandtoday', lambda x: x.dt.days // 7),
                                          frequency=('total_order', 'sum'),
                                          monetary_cltv_avg=('total_purch', 'mean'))
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],cltv_df['monetary_cltv_avg'])
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,  # 6 aylık
                                       freq='W',  # T'nin frekans bilgisi
                                       discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv["segment"] = pd.qcut(cltv["clv"], 4, labels=["D", "C", "B", "A"])
    return dataframe



























