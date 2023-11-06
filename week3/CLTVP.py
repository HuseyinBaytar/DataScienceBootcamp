# Müşteri yaşam Boyu değeri tahmini

#zaman projeksiyonlu olasılıksal lifetime value tahmini

#satın alma sayısı  * satın alma başına ortalama kazanç


#CLTV = ( customer VAlue /churn rate) x profit margin
#Customer value = purchase frequency  * average order value
#CLTV = expected number of transaction * expected average profit

#bütün kitlenin satın alma davranışlarını bir olasılık dağılımı ile modellicez daha sonra bu olasılık dağılımı ile modellediğimiz davranış biçimlerini KOŞULLU yani kişi özelinde biçimlendirecek şekilde her bir kişi için beklenen satın alma/işlem sayılarını tahmin edicez.

#bütün kitlemizin average profit değerini olasılıksal olarak modellicez, daha sonra bu modeli kullanarak kişi özelliklerini girdiğimizde, kişi özelinde conditional expected average profitlerini koşullayarak ana kitlenin dağılımından beslenmiş bir biçimde her bir kişi için average profitleri hesaplıcaz


# CLTV =  BG/NBD modeli   *  Gamma gamma submodel

##############################################################
#bg/ndb  ile expected number of transaction
##############################################################
#bg/ndb modeli, Expected Number of Transaction için iki süreci olasılıksal olarak modeller.
#transaction PRocess (buy) ve dropout protecss (till you die)
#bg/ndb nami diğer: Buy Till You Die

#transaction process(buy):
#Alive olduğu sürece, belirli bir zaman periyodunda, bir müşteri tarafından gerçekleştirelecek işlem sayısı transaction rate parametresi ile possion dağılır.
#daha basit olarak bir şekilde;  Bir müşteri alive olduğu sürece kendi transaction rate'i etrafında rastgele satın alma yapmaya devam edecektir.

#transaction rate'ler her bir müşteriye göre değişir ve tüm kitle için gamma dağılır.

#   Dropout process (till you die)
# her bir müşterinin p olasılığı ile dropout rate'i vardır.
# bir müşteri alışveriş yaptıktan sonra belirli bir olasılıkla drop olur.

#dropout rateler her bir müşteriye göre değişir  ve tüm kitle için beta dağılır.


##### yani  transaction rate = gamma dağılım //  dropout rate = beta dağılır ##########

################ Gamma Gamma Sub model
#Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
#bir müşterinin işlemlerinin parasal değeri transaction valuelarının ortalaması etrafında rastgele dağılır.
#ortalama transaction value, zaman içinde değişebilir ama tek bir kullanıcı için değişmez.
#ortalama transaction value tüm müşteriler arasında gamma dağılır.


############################################################################################
############# BG-NBD ve Gamma-Gamma ile CLTV Prediction ################
############################################################################################
#1- Verinin Hazırlanması
############################################################################################

# InvoiceNO: Fatura numarasi.eşsiz numara, C ile başlıyorsa iptal edilen işlem
# StockCode: ürün kodu. Eşssiz numara
# Description : ürün ismi
# Quantity: Ürün adedi. faturalardaki ürünlerden kaçar tane satıldığını ifade eder.
# InvoiceDate: faturanin tarihi ve zamani
# UnitPrice: Ürün fiyatı
# customerID: eşssiz müşteri numarası
# Country: ülke ismi

#pip install lifetimes
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)   #bütün sütunları gör
pd.set_option('display.max_rows', 500)     #bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.4f' % x)    #virgülden sonra kaç tane sayı görmek istiyorsak

df_ = pd.read_excel(r"C:\Users\Curbe\Desktop\datascience\bootcamp\week3\online_retail_II.xlsx" ,sheet_name="Year 2010-2011")
df = df_.copy()


def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 +1.5 * interquantile_range
    low_limit= quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable]  = up_limit


df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"]> 0]
df = df[df["Price"]> 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

############################################################################################
#                          Lifetime Veri yapısının HAzırlanması
#recency: son satın alma üzerinden geçen zaman.  haftalık.
#T: Müşterinin yaşı. HAftalık.( analiz tarihinden ne kadar süre önce ilk satın alma ypaılmış)
#frequency: tekrar eden toplam satın alma sayısı (frequency >1 )
#monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                         lambda x: (today_date - x.min()).days],
                                         'Invoice': lambda x: x.nunique(),
                                         'TotalPrice': lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency","monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7



############################################################################################
#2- BG-NBD Modeli ile Expected Number of Transaction
############################################################################################
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"]= bgf.predict(1,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T']).sort_values(ascending=False)

#1 ay içinde en çok satın alma beklediğimiz müşteri kimdir?
bgf.predict(4,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"]= bgf.predict(4,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T']).sort_values(ascending=False)


bgf.predict(4,
                cltv_df['frequency'],
                cltv_df['recency'],
                cltv_df['T']).sum()

#3 ayda tüm şirketlerin beklenen satış sayısı nedir?

bgf.predict(4* 3,
                cltv_df['frequency'],
                cltv_df['recency'],
                cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4* 3,
                cltv_df['frequency'],
                cltv_df['recency'],
                cltv_df['T'])

#tahmin sonuçlarının değerlendirilmesi

plot_period_transactions(bgf)
plt.show()


############################################################################################
#3- Gamma-Gamma Modeli ile Expected Average Profit
############################################################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)



cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)


############################################################################################
#4-BG-NBD ve Gamma-Gamma modeli ile CLTV'nin Hesaplanması
############################################################################################
cltv = ggf.customer_lifetime_value(bgf,
                                      cltv_df['frequency'],
                                      cltv_df['recency'],
                                      cltv_df['T'],
                                      cltv_df['monetary'],
                                      time= 3, #3 aylık
                                      freq='W', #T'nin frekans bilgisi
                                      discount_rate=0.01 )

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on='Customer ID',how='left')

cltv_final.sort_values(by='clv', ascending=False).head(10)

#eğer müşterin dropout olmadıysa, recency'si arttıkça satın alma olasılığı yükseliyor der.
#yani birşey satın aldı, satın alma ihtiyacını bitirdi fakat belirli bir zaman geçtikten sonra tekrardan satın alma ihtiyacı ortaya çıkar


############################################################################################
#5-CLTV'ye Göre Segmentlerin Oluşturulması
############################################################################################
cltv_final["segment"] = pd.qcut(cltv_final["clv"],4, labels=["D","C","B","A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count","mean","sum"})



############################################################################################
#6- fonksiyonlaştırma
############################################################################################

def create_cltv_p(dataframe,month=3):
    dataframe.dropna(inplace=True)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    replace_with_thresholds(df, "Quantity")
    replace_with_thresholds(df, "Price")
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                             lambda x: (today_date - x.min()).days],
                                             'Invoice': lambda x: x.nunique(),
                                             'TotalPrice': lambda x: x.sum()})
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=3,  # 3 aylık
                                       freq='W',  # T'nin frekans bilgisi
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on='Customer ID', how='left')
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
    return cltv_final















































