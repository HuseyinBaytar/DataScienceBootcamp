#                                   Customer Lifetime Value(Müşteri Yaşam Boyu Değeri)

# Bir müşterinin bir şirketle kurduğu ilişki-iletişim süresince bu şirkete kazandıracağı parasal değerdir.
#CLTV değerini hesaplamak ve tahmin etmek farklı şeylerdir

# Nasıl hesaplanır?

#satın alma başına ortalama kazanç *  satın alma sayısı     = müşteri potansiyel değer hesabı

# CLTV = (customer value /churn rate) x profit margin

#Customer value = average order value * purchase frequency
#Average order value =  total price / total transaction
#purchase frequency = Total transaction / Total number of Customers
# Churn rate = 1 - repeat rate
#Repeat rate = Birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler
#profit margin = total price * 0.10

#sonuç olarak her bir müşteri için hesaplanacak olan CLTV değerlerine göre bir sıralama yapıldığında  ve CLTV değerlerine göre belirli noktalardan bölme işlemi yapılarak gruplar oluşturulduğunda müşterilerimiz segmentlere ayrılmış olacaktır.


######################################################################################
# CUSTOMER LIFETIME VALUE (MÜŞTERİ YAŞAM BOYU DEĞERİ)
######################################################################################

# 1. Veri Hazırlama
######################################################################################
# Değişkenler

# InvoiceNO: Fatura numarasi.eşsiz numara, C ile başlıyorsa iptal edilen işlem
# StockCode: ürün kodu. Eşssiz numara
# Description : ürün ismi
# Quantity: Ürün adedi. faturalardaki ürünlerden kaçar tane satıldığını ifade eder.
# InvoiceDate: faturanin tarihi ve zamani
# UnitPrice: Ürün fiyatı
# customerID: eşssiz müşteri numarası
# Country: ülke ismi


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)   #bütün sütunları gör
#pd.set_option('display.max_rows', None)     #bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.5f' % x)    #virgülden sonra kaç tane sayı görmek istiyorsak

df_ = pd.read_excel(r"C:\Users\Curbe\Desktop\datascience\bootcamp\week3\online_retail_II.xlsx" ,sheet_name="Year 2009-2010")
df = df_.copy()

df.head()
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]
# birden fazla alışveriş yapan ID'lerin faturalarını tek id'de birleştirdik
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

#1. total transaction -  eşssiz fatura sayısı değişkeni
#2. toplam kaç birim satın almış
#3. toplam hesap

cltv_c.columns = ['total_transaction','total_unit','total_price']


######################################################################################
#2. Average Order Value ( average_order_value = total_price/ total_transaction)
######################################################################################

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]



######################################################################################
#3. Purchase Frequency (total_transaction / total_number_of_customers)
######################################################################################

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]


######################################################################################
#4. Repeat RAte and Churn RAte  ( birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
######################################################################################

repeat_rate = cltv_c[cltv_c["total_transaction"]>1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate


######################################################################################
#5. Profit Margin (profit_margin = total_price * 0.10)
######################################################################################

cltv_c['profit_margin']= cltv_c['total_price'] * 0.10



######################################################################################
#6. Customer Value ( customer_value = average_order_value * purchase_frequency)
######################################################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c['purchase_frequency']



######################################################################################
#7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
######################################################################################

cltv_c["cltv"] = (cltv_c["customer_value"]/ churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv",ascending=False ).head()


######################################################################################
#8. Segmentlerin Oluşturulması
######################################################################################

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D","C","B","A"])

cltv_c.sort_values(by="cltv",ascending=False ).head()

cltv_c.groupby("segment").agg({"count","mean","sum"})


#çıktı almak için
#cltv_c.to_csv("cltv_c.csv")


######################################################################################
#9. Bonus : Tek fonksiyon
######################################################################################

def create_cltv_c(df, profit=0.10):

    df = df[~df["Invoice"].str.contains("C", na=False)]

    df = df[(df['Quantity'] > 0)]
    df.dropna(inplace=True)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                            'Quantity': lambda x: x.sum(),
                                            'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10
    cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c['purchase_frequency']
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_c


