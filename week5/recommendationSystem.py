# kullanıcılara bazı teknikleri kullanarak, ürün yada hizmet önermek şeklindeki sistemlerdir.

#içerik çok bol, kullanıcının ilgi alanı daha az bi içerik setine yönelik, dolayısıyla bu çok bol olan içerikleri filtrelememiz lazım.
#alt kümelere indirmemiz lazım

#bol olan hizmet, ürün, içerik vs filtrelemek  asıl amacımız.

###### Simple Recommender Systems (basit tavsiye sistemleri)
# - iş bilgisi ya da basit tekniklerle yapılan genel öneriler.
# - Kategorinin en yüksek puanlılarını, trend olanları, efsaneleri vs önermek, mantık barındırmadan iş bilgisiyle gerçeleştirilecek olan basit sistemdir.

###### Assosication Rule Learning (birliktelik Kuralı öğrenimi)
# - birliktelik analizi ile öğrenilen kurallara göre ürün önerileri.
# - sepet analizi olarakta karşımıza gelebilir. offline/online bir çok şirket kullanır. e-ticarette yaygınca kullanılır
# - çok sık bir şekilde birlikte satın alınan ürünlerin olasılıklarını çıkarır ve bunlara göre belirli öneriler yapar


###### Content Based Filtering ( içerik temelli filtreleme)
# - ürünlerin benzerliğine göre önerilen yapılan uzaklık temelli yöntemlerdir.

######  Collaborative Filtering ( işbirlikçi filtreleme)
# - Topluluğun kullanıcı ya da ürün bazında ortak kanaatlerini yansıtan yöntemlerdir
# - user based /  item based / model based (matrix factorization)



##############################################################################################################
# Birliktelik kuralı öğrenimi (Association Rule Learning)

#  veri içerisindeki örüntüleri (pattern,ilişki,yapı) bulmak için kullanılan birliktelik kuralları tabanlı bir makine öğrenmesi tekniğidir.

#örn: wallmartta bebek bezi alanların bira aldığını görüyoruz, wallmart marketlerini dizayn ederken bebek bezi ile birayi yan yana reyonlara koyup büyük bir kar elde ediyor.

#örn: bir markette yapılan 8 alışverişin fişivar elimizde. 5 tanesinde ekmek ve süt aynı anda alınmış,  5/8, bu incelenen fişler üzerine; alışverişlerin yaklaşık olarak %62'sinde ekmek ve süt birlikte görüntülenmiştir deriz.  birliktelik kuralı örneğidir


### Apriori Algoritması
# sepet analizi yöntemidir. Ürün birlikteliklerini ortaya çıkarmak için kullanır.

# bizim için değerli olan 3 metriği vardır,
# 1.si support değeri x ve y'nin birlikte görülme olasılığıdır   freq(X,Y)/N    5 ekmeksüt/ 8 totalişlem  = 5/8
# 2. Confidence X satın alındığında y'nin satılma oranı   freq(x,y) / freq(x)
# 3. Lift X satın alındığında Y'nin satın alınma olasılığı lift kat kadar artar.   support(x,y) / (Support(x) * Support(y))

#bu 3 metrik üzerinden bize çok değerli istatistiksel değerler verir.

#kullanıcı davranışı = satın alması veya almaması.

### apriori nasıl çalışır?
# apriori algoritması adım adım çalışmanın başında belirlenecek bir support eşik değerine göre, olası ürün çiftlerini hesaplar ve her iterasyonda belirlenen support değerine göre elemeler yaparak nihai final tablosunu oluşturur.

# N: işlem sayısı
# support: eşik değer diyelim %20(0,2) eğer  bir ürün %20'den az gözlemleniyosa bizim için önemli değildir diye bir eşik değer atıyoruz

#adım1: her ürünün supportunu hesapla,
# her ürünün toplam frekansını hesaplarız. ve support değeri için freq(X,Y)/N  diyelim ki 2 kere geçiyor 5 değerde, 0.4 olur sup değeri

#adım2: supporta göre eleme yap
# eşik değerini geçemeyenleri eleriz.

#adım3: yeni liste ve support'u oluştur.
#süt yumurta / süt çay / süt kola,  yumurta/çay yumurta /kola, tarzı  listenin içindeki her ürünün diğer ürünlerle tek tek yazılması
#ilk listeye bakıp, 2. yeni listedeki supportları karşılaştır

#adım 4. supporta göre eleme yaparız

#adım 5. yeni liste ve support'u oluştur
#süt/çay/yumurta,   süt/çay/yumurta/kola,   süt/çay/kola   gibi  toplu bir liste yaparız bu sefer


#adım 6. supporta göre eleme
#frekansları yine %20'den düşük olanları eleriz

#adım7: final tablosu oluşturup yorumlarız,
#freq/support/confidence/lift olarak


###################################
#proje: birliktelik kuralı temelli tavsiye sistemi

#iş problemi: sepet aşamsındaki kullanıcılara ürün önerisinde bulunmak.
#veri seti: online retail II

# features
#invoiceNO: fatura numarası, eğer C ile başlıyosa iptal
#stockcode : ürün kodu
#description : ürün ismi
#quantitiy: ürün adedi
#invoicdedate: fatura tarihi
#unitprice: fatura fiyatı
#customerid: eşsiz müşteri numarası
#country: ülke ismi

###### 1. veri ön işleme

#!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("week5/datasets/online_retail_II.xlsx",sheet_name="Year 2010-2011")
#klon oluşturuyoruz, tekrar yüklememek için
df = df_.copy()
df.head()

# invoice'daki C'ler geri iadeyi kast eder, o yüzden - döndürüyor minde
df.describe().T
#eksik gözlemler var
df.isnull().sum()
#541.910 tane var
df.shape



def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)                                                # boş değerleri düşür
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]      # içinde C geçen iade faturaları çıkar
    dataframe = dataframe[dataframe["Quantity"] > 0]                              # quantitiy değeri 0'dan büyükleri al
    dataframe = dataframe[dataframe["Price"] > 0]                                 # price değeri 0 'dan büyükleri al
    return dataframe

df = retail_data_prep(df)

#aykırı değerleri atmak için fonksiyon, aykırı değer : bir değişkendeki genel dağılımın dışında olan değerlerdir.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)           # girdiğim değişkenin %1 çeyrek değerini hesapla q1 olarak tut  // %25 de yapılabilir
    quartile3 = dataframe[variable].quantile(0.99)           # %99 çeyrek değerini hesapla q3 olarak tut                     // %75 de yapılabilir
    interquantile_range = quartile3 - quartile1              # üst çeyrek değerden alt çeyrek değeri çıkardık ve iqr değeri geldi
    up_limit = quartile3 + 1.5 * interquantile_range         # 99'luk çeyrek değerden 1.5 iqr uzaklıktaki nokta benim üst limitimdir
    low_limit = quartile1 - 1.5 * interquantile_range        # 1'lik çeyrek değerden 1.5 iqr uzaklıktaki nokta benim alt limitimdir
    return low_limit, up_limit                               # alt ve üst limiti döndür

def replace_with_thresholds(dataframe, variable):                                      #yukarda hesapladığımız üst ve alt değerlere outlierları baskılamak için yazılan fonksiyon
    low_limit, up_limit = outlier_thresholds(dataframe, variable)                      # threshold fonksiyonundaki üst ve alt limitlerini çağırıyoruz.
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit             # df'de ilgili değişkenin indexine erişip, low limitten aşşağıda olanları, low limitle değiştir
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit               # df'de ilgili değişkenin indexine erişip, up limitten yukarı olanları, up limitle değiştir



def allinone(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = allinone(df)


df.describe().T

############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################
#daha ölçülebilir, üzerinde işlemler yapılabilir bir matrix yapısı istiyoruz.
#invoicelar sepet olacak bizim için // satırsa sepetler, sütunlarda tüm ürünler olmalı

#sadece fransa müşterilerine bakıyoruz, veri setimizi sadece fransadakiler olsun diye bu kodu yazdık.
df_fr = df[df['Country'] == "France"]

#normalde stockcode'a göre gidice ama gözlemlemek için descriptionu alıyoruz
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

#groupby'a al invoce'a ve descp' e göre, quantitynin sumunu al her faturadaki her üründen kaçtane olduğu bilgisi ile unstack olarak pivot ediyoruz
#index based seçim yapıp 5 satır 5 sütun getir diyoruz ama eksik değerler NaN gözüküyor
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

#unstack işleminden sonra boş olan yerleri 0 ile doldurduk
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

#24 yazan yerde 1 yazması lazım, yani boş olanlar 0, diğerleri 1 olacak şekilde, olacak
df_fr.groupby(['Invoice', 'StockCode']). \
              agg({"Quantity": "sum"}). \
              unstack(). \
              fillna(0). \
              applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
#applymap ile tüm gözlemleri gezer


def create_invoice_product_df(dataframe, id=False):
    if id:                                                                                               # eğer id true ise
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)                                                        #stock koda göre yukardaki işlemi yapıyoruz
    else:                                                                                                # eğer id false ise
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)                                                        #tekrardan yukarda yaptığımız işlemler

fr_inv_pro_df = create_invoice_product_df(df_fr)   #descriptiona göre

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)  #id'lere göre


def check_id(dataframe, stock_code):                                                                        #check id  diye bir fonksiyon tanımlıyoruz
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()      # girilen df'den stockcode seçilecek ve sorgulamak istenen stock id'nin dscription gelecek
    print(product_name)


check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################
# appriori fonksiyonu methodu ile olası tüm ürün birliktellerin support değerlerini bulmak  1. önceliğimiz

frequent_itemsets = apriori(fr_inv_pro_df,             #df veriyoruz
                            min_support=0.01,          # min support argümanına 0.01 girdik
                            use_colnames=True)
#olası kombinasyonları çıkardık, her bir ürünün olasılığı
frequent_itemsets.sort_values("support", ascending=False)

#birliktelik kuralarrını çıkarıyoruz
rules = association_rules(frequent_itemsets,   #associationrules methodu ile, az önce oluşturduğumuz frequen_itemsets'i giriyoruz
                          metric="support",    #metrik olarak supportu giriyoruz
                          min_threshold=0.01)  #thresholdumuzu girdik
#leverage: kaldıraç etkisi demek, supportu yüksek olan değerlere yüksek verir
#lift : daha az sıklıkta olmasına rağmen bazı ilişkileri yakalıyabilir, daha değerlidir bizim için, yansız metriktir.
#conviction: y olmadan x ürünün beklenen değerinin frekansıdır.

#support: x ürünü ile y ürününün birlikte satın alınma olasılığı
#confidence: x ürünü satın alındığında y ürününün satın alınma olasılığı
# lift : x ürünü alınca y ürününün satın alınma olasılığı  ....  artmaktadır

#örnek olarak içinden supportu 0.05'den yüksek, confidenci 0.1'den büyük ve, lifti 5'denbüyük olanları aldık
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
#84 ürüne kadar indirgendi

#ürünün ismine bakıyoruz
check_id(df_fr, 21086)


#yukarda aldığımızı sort values ile confidence'a göre yüksekden düşüğe azalan şekilde sıraladık
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################
#daha öncesinden gerçekleşebiliecek olası senaryolara karşı kime yada hangi ürüne neyi önerebiliecğeimizi bir tabloda tutarz.
#kullanıcı siteye girip sepete bişi eklediği an hazır olan yerden direkt çekeriz

# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)
#bu senaryoda kafamıza göre lift'e göre sıraladık, aynı şekilde confidence vs sıralanabilir. yoruma kalmış.
sorted_rules = rules.sort_values("lift", ascending=False)

#antecedents bölümünde gezicez.burada yakalamış olduğumuz ürünleri, aynı indexdeki diğer kısımdaki consequentste gördüklerimizi yazacaz
#lifte göre sıralanmış olduğu için, diyelim ki antecedenste 5 numaralı ürünü gördük, indexine bakıyoruz 1666, 1666'nın consequentsteki değerini yazdırıcaz
#birden fazla ürün olursa diye liste oluşturduk
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):                                # productaki tüm satırları gez, index bilgileriyle birlikte, "antecedenst"'e göre sıralı
    for j in list(product):                                                              # set yapısında belirli bir işlem yapabilmemiz için, listeye çeviriyoruz  list(product) ile
        if j == product_id:                                                              # bu listenin içindede gezmek için tekrar bir for döngüsü yazıyoruz, bu elemanlarda geziyoruz
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])     # gezdiğimiz elemanlarda eğer product id yakalarsan, o değerin indexindeki "consequents"'i listeye ekle
                                                                                         # değer olarak getirmesini istediğimiz için, [0] ile ilk gördüğünü getir diyoruz
recommendation_list[0:10]

check_id(df, 22728)



#bir fonksiyonda özetliyoruz
def arl_recommender(rules_df, product_id, rec_count=1):                               #rec count diye bir argüman girdik onun dışındaki işlemler aynı, kuralları, product id(stock code)'yi giriyoruz
    sorted_rules = rules_df.sort_values("lift", ascending=False)                      #ruleları lifte göre sıraladık, yorum confidence sıralanabilir
    recommendation_list = []                                                          #boş bir liste oluşturduk
    for i, product in enumerate(sorted_rules["antecedents"]):                         #yukardaki işlemi gerçekleştirip, listeye ürünleri atıyoruz
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)  #1 ürün önerir
arl_recommender(rules, 22492, 2)  #2 ürün önerir
arl_recommender(rules, 22492, 3)  #3 ürün önerir



def arl_recommender(rules_df, product_id, rec=1):
    #olması gereken
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id :
                for k in list(sorted_rules.iloc[i]["consequents"]):
                    if k not in recommendation_list:
                        recommendation_list.append(k)

    return recommendation_list[0:rec]
