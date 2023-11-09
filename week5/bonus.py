#Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane yada 1'den fazlaolabilir.
# Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.Kullanıcı1’in sepetinde bulunan
# ürünün id'si: 21987       Kullanıcı2’in sepetinde bulunan ürününid'si: 23235           Kullanıcı3’in sepetinde bulunan ürünün id'si: 22747

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#Veri seti Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009-09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.Şirketin ürün kataloğunda hediyelik
# eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.


####################################################Görev 1: Veriyi Hazırlama

#Adım 1: Online RetailII veri setinden 2010-2011 sheet’ini okutunuz.


df_ = pd.read_excel(r"C:\Users\Curbe\Desktop\datascience\bootcamp\week3\online_retail_II.xlsx" ,sheet_name="Year 2010-2011")
df = df_.copy()
df = df[df['Country'] == "Germany"]
df.head()

# Adım 2: StockCode’u POST olan  gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df[df["StockCode"] == "POST"]

df = df[df["StockCode"] != "POST"]

# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df = df.dropna()

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].str.contains("C", na=False)]

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df = df[(df['Price'] > 0)]


# Adım 6: Priceve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
df.describe([0.01,0.99]).T
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

replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
#########################Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme

#Adım 1:Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız

def create_invoice_product_df(dataframe, id=False):
    if id:                                                                                               # eğer id true ise
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)                                                        #stock koda göre yukardaki işlemi yapıyoruz
    else:                                                                                                # eğer id false ise
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)                                                        #tekrardan yukarda yaptığımız işlemler

gr_inv_pro_df = create_invoice_product_df(df)   #descriptiona göre

gr_inv_pro_df = create_invoice_product_df(df, id=True)  #id'lere göre





#Adım 2: Kuralları oluşturacak create_rulesfonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df, id=True, country="Germany")

rules.head()

################################Görev 3: Sepet İçerisindeki Ürün Id’leriVerilen Kullanıcılara Ürün Önerisinde Bulunma
#  Adım 1:check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 21987)
#['PACK OF 6 SKULL PAPER CUPS']

check_id(df, 23235)
#['STORAGE TIN VINTAGE LEAF']

check_id(df, 22747)
#["POPPY'S PLAYHOUSE BATHROOM"]

#  Adım 2: arl_recommenderfonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 1)
arl_recommender(rules, 22747, 1)



#  Adım 3: Önerilecek ürünlerin isimlerine bakınız.
check_id(df, 21989)
#['PACK OF 20 SKULL PAPER NAPKINS']

check_id(df, 23244)
#['ROUND STORAGE TIN VINTAGE LEAF']

check_id(df, 22746)
#["POPPY'S PLAYHOUSE LIVINGROOM "]
