#RFM nedir =   Recency, Frequency, Monetary    baş harfleri = RFM
#rfm analizi müşteri segmentasyonu için kullanılan bir teknik
#müşterilerin satın alma alışkanlıkları üzerinden gruplara ayrılması ve bu gruplar özelinde stratejiler geliştirebilmesini sağlar
#CRM çalışmaları için bir çok başlıkta veriye dayalı aksiyon alma imkanı sağlar

#               RFM Metrikleri
#  Recency(yenilik) : müşterinin yenilik, yada bizden en son ne zaman alışveriş yaptığı durumunu ifade eder. örneğin bir müşterinin değeri 1, diğerinin 10 ise, 1 olan bizim için daha iyidir, henüz bir gün önce alışveriş yaptığını ifade eder eğer günlük cinsten konuşuyorsak

#Frequency(Sıklık) : müşterinin yaptığı alışveriş/işlem sayısı.

#Monetary(Parasal Değer):  Müşterilerin bize bıraktığı parasal değeri ifade eder


#recency için en iyi değer küçük olan değer, Frequency için en iyi değer en büyük olan, Monetarcy içinde en yüksek değer en iyidir yorumunu yapabilriz.


#               RFM Skorları
# rfm metriklerini rfm skorlarına çevirmemiz lazım, hepsini aynı cinsten ifade edip karşılaştırma yapmak için.

#R / F / M   değerlerinden gelenleri RFM'e getiriyoruz
#1 / 4 / 5   =  145


#RFM skorları üzerinden segmentler oluşturmak gerekiyor.

#       RFM ile Müşteri Segmentasyonu ( Customer Segmentation with RFM)

###############################################################################
#                   1. İş Problemi (bussiness problem)
###############################################################################


#bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejilerini belirlemek istiyor.
#online retail II isimli bir veri seti İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içerir.

# Değişkenler

# InvoiceNO: Fatura numarasi.eşsiz numara, C ile başlıyorsa iptal edilen işlem
# StockCode: ürün kodu. Eşssiz numara
# Description : ürün ismi
# Quantity: Ürün adedi. faturalardaki ürünlerden kaçar tane satıldığını ifade eder.
# InvoiceDate: faturanin tarihi ve zamani
# UnitPrice: Ürün fiyatı
# customerID: eşssiz müşteri numarası
# Country: ülke ismi

###############################################################################
#           2. Veriyi anlama ( DAta understanding)
###############################################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)   #bütün sütunları gör
pd.set_option('display.max_rows', None)     #bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.5f' % x)    #virgülden sonra kaç tane sayı görmek istiyorsak

df_ = pd.read_excel(r"C:\Users\Curbe\Desktop\datascience\bootcamp\week3\online_retail_II.xlsx" ,sheet_name="Year 2009-2010")
df = df_.copy()

df.head()
df.shape
df.isnull().sum()

#essiz urun sayısı nedir?
df["Description"].nunique()

#hangi üründen kaçar tane var?  #bu eşsiz ürünler kaçar defa faturaya gündem oldu?
df["Description"].value_counts().head()

#en çok sipariş edilen ürün hangisi? #peki  bu eşssiz ürünlerden kaçar tane satıldı?
df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

#kaç adet fatura kesildi?
df["Invoice"].nunique()

#fatura başına toplam kaç para kazanılmıştır?
#ürünlerin kaçar tane satıldığı ile fiyatını çarpıp toplam fiyatı bulduk
df["TotalPrice"] = df["Quantity"] * df["Price"]
#fatura başına toplam kaç para ödendiğini hesaplamak için, faturayı groupby'a alıp, totalprice'ın toplamını alırız.
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()



###############################################################################
#               3. Veri Hazırlama( Data Preparation)
###############################################################################
df.shape
df.isnull().sum()
#customer id'ler eksik olduğundan dolayı onları silebiliriz, customer segmentation amacımız, customer id yok ise segmente edemeyiz.
df.dropna(inplace=True)   #boş olanları df'imizden attık

#invoce'da başında C olan ifadeler, iadeleri ifade etmektedir.
df.describe().T
#fiyat - olamıyacağından dolayı bunlar iadelerden kaynaklanmaktadır. -'leri df'den çıkarmamız lazım
df = df[~df["Invoice"].str.contains("C", na=False)]


###############################################################################
#            4.RFM Metriklerinin hesaplanması(calculating RFM Metrics)
###############################################################################

#Recency, Frequency, Monetary
df.head()
#veri seti 2010'lara dayandığı için ve biz o tarihte olmadığımız için, sanki 2 gün sonra analiz ediyormuş gibi yapmamız lazım
df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)
type(today_date)

#Recencyi bulmak için today date'den groupbya aldıktan sonra her bir müşterinin max tarihini bulmamız lazım ve today date'den çıkardınca Recency buluruz.

#Frequency için customerID'ye göre groupbya aldıktan sonra her bir müşterinin eşsiz fatura sayısına gidersek, müşterinin kaç tane satın alma yaptığını buluruz

#Monetary için customerID'ye göre groupby yaptıktan sonra totalprice'ların sum'unu alırsak, her bir müşterinin kaç para bıraktığını buluruz

rfm = df.groupby("Customer ID").agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice' : lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.describe().T

rfm = rfm[rfm["Monetary"] > 0 ]

rfm.shape


###############################################################################
#          5.RFM Skorlarının Hesaplanması (Calculating RFM Score)
###############################################################################

#Recency = ters
#freq ve monetary = düz
#yani büyük puan kötü sonuç recency,  büyük puan iyi sonuç freq ve monetary

rfm["recency_score"] = pd.qcut(rfm['Recency'],5, labels=[5,4,3,2,1])

# 0-100 arası , 0-20, 20-40, 40-60, 60-80, 80-100 gibi 5 e böler qcut 5 dersek
#iyi olan 1, kötü olan 5 olacak şekilde puanlama yaptık recenyde

rfm["monetary_score"] = pd.qcut(rfm['Monetary'],5,labels= [1,2,3,4,5])
#iyi olan 5 kötü olan 1
rfm.head()

#büyük olan değerlidir frequencyde,  rank methodu first olarak kullandık çünkü benzer değerler var , ilk gördüğünü al dedik
rfm["frequency_score"]= pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm.head()

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


rfm.describe().T


rfm[rfm["RFM_SCORE"]== "55"]


###############################################################################
#       6.RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi
###############################################################################

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

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


rfm.head()

rfm[["segment","Recency","Frequency","Monetary"]]. groupby("segment").agg(["mean","count"])



rfm[rfm["segment"]== "cant_loose"].head()

#cant_loose'un id'leri gelir
rfm[rfm["segment"]== "cant_loose"].index


new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# new_df.to_csv("new_customers.csv")

#bu id'ler yeni müşteriler, kendileri gelmiş vs gibisinden ilgili departmana verilir

#rfm.to_csv("rfm.csv")
#rfm'i dışarı çıkarır.




###############################################################################
#          7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################################

def create_rfm(dataframe):

    #VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    #RFM Metriklerinin hesaplanması
    today_date = dt.datetime(2011,12,11)
    rfm = dataframe.groupby("Customer ID").agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
    rfm.columns = ['recency','frequency','monetary']
    rfm= rfm[(rfm['monetary']> 0 )]

    #RFM skorlarının hesaplanması
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    #cltv_df skorları kategorik değere dönüşüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    #segmentlerin isimlendirilmesi
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
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency","frequency","monetary","segment"]]
    rfm.index = rfm.index.astype(int)
    return rfm

df = df_.copy()

rfm_new = create_rfm(df)


