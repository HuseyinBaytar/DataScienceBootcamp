# sıralama konusu sadece ürünler için değil bir çok işlemde karşımıza gelebilicek bir durumdur.

#örneğin işe alım,  mülakat puanı, ingilizce puanı, teknik dil puanı  hepsine farklı ağırlık koyup, ortalaması üzerinden bir sıralama yapılabilir
#örnerğin bir siteden  alışveriş yapmadan önce arama yerine yazılan keyworde en uygun olanları getirmesi için.


## uygulama: kurs sıralama

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)

df = pd.read_csv("week4/product_sorting.csv")

df.shape
df.head()


# sorting by rating

df.sort_values("rating",ascending=False).head(20)

#sadece ratinge göre sıralama yapmamalıyız, mesela comment countu çok düşük ama puanı yüksek kurslar var
#hem satın alma sayısı/ hem puan sayısı / hemde yorum sayısını aynı anda göz önüne alacak bi sınıflandırma gerekir.


# Sorting by Comment Count or Purchase Count

#satın alma ya göre sıraladık ama satın almaya göre sıraladığımızda belki kötü kurslarıda yukarı çıkardık, yani çok satın alınmış ama kötü kurs olabilir
#kullanıcı memnuniyetinide işin içine katmamız lazım.
df.sort_values("purchase_count",ascending=False).head(20)


#yorum sıralarına göre sıraladığımızda, puan sıralarına göre daha güzel geldi gibi ama yine benzer problemler var, örneğin satın alması yüksek ama ücretsiz bir şekilde dağıtılmış olabilir.
df.sort_values("commment_count",ascending=False).head(20)


#####################################################################
#sorting by rating, comment and purchase
#Derecelendirme, SAtın alma, yoruma göre sıralama
#####################################################################
# satın alma frekanslarını 1-5 e baskıladık
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])

# comment countu 1 ile 5 arasında baskıladık
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["commment_count"]]).transform(df[["commment_count"]])


df.describe().T

#ağırlıklar kişisel olara kverilebilir, rating en önemli olduğu için %42 verdik fakat değişebilir
#satın almaya %26 verdik ama mesela satın alınan kursun dışında bedava hediye verilmiş olabilir.
#commente göre %32, satın alıma göre %26, ratinge göre %42 ağırlık vererek skorları hesapladık.

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"]* 26 / 100 +
 df["rating"] * 42 /100)

#sosyal ispati görmek için ağırlıklar vererek bakıyoruz.

#tek fonksiyon;
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"]* w2 / 100 +
            dataframe["rating"] * w3 /100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

#3 tane faktörün ağırlıklarını biçimlendirerek, sağlıklı bir sıralama yaptık.
df.sort_values("weighted_sorting_score", ascending=False).head(20)


#eğer ilgisiz olanları çıkarmak istiyosak, örneğin veri bilimi için araştırıyoruz, içinde veri bilimi geçmiyenleri çıkaracaksak
# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)



###################################
#Bayesian Average Rating Score

#sorting products with 5 star rated
#sorting products according to distribution of 5 star rating
import math
import scipy.stats as st


#puan dağılımlarının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar.
#veri setindeki 5_point, 4_point, 3_point, 2_point, 1_point'i vermemiz lazım
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1) * (n[k]+ 1) / (N + K)
        second_part += (k +1 ) * (k +1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


#bar sorting score,  bar rating, bar average rating diyede isimlendirilebilir.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

#eğer bütün satın almalar ve yorumlarla ilgili bir refereans noktam olsaydı ve bunlara tam olarak güveniyor olsaydım, bilimsel olarak gelebilicek en doğru, en tutarlı sıralamayı bayes yöntemiyle elde etmiş olacaktık.

df.sort_values("bar_score", ascending=False).head(20)

#kullanılabilir fakat daha sonra başka işlemlere sokacağımızdan dolayı, birazda dağılım bilgisine referansla oluşan değerler olduğundan dolayı, skor da diyebiliriz.

# bar score bize sadece ratinglere odaklanarak bir sıralama sağladı dolayısıyla ratelerin dağılımına bakarak oluşturulmuştur diğer birden fazla değere bakarak yaptığımız weighted_sorting_score'a göre farklı çıkacaktır, bar score'da kullanılabilir, weighted sorting score'da kullanılabilir.



#course_1'in daha yukarda olmasının sebebi çok daha az sayıda puana sahip olduğu halde, düşük puan miktarları, diğer kursa göre daha az olmasından dolayı yukarı çıkmıştır.
df[df["course_name"].index.isin([5,1])].sort_values("bar_score", ascending=False).head(20)



###############################################
#karma sıralama //   hybrid sorting
###############################################

#bar score ile diğer faktörleri birleştirip yapılan bir sıralamadır.

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]),axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score*bar_w/100 + wss_score*wss_w/100


hybrid_sorting_score(df)
#bilimsel, iş bilgisi hemde yeni potansiyel yıldızlarada şans veren bir şekilde sıralama işlemi gerçekleşti
#bar score'a %60 vererek, potansiyel vaad edenleri yakaladık.
df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(5)


#########################################################################################
# uygulama : IMDB Movie Scoring & Sorting
####################################################################################

import pandas as pd
import math
import scipy.stats as st

df = pd.read_csv("week4/movies_metadata.csv", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

######################
# vote average'a göre sıralama
#vote countu 1 olanları gösterdi
df.sort_values("vote_average", ascending=False).head(10)


#çeyrek değerlere göre filmlerin aldıkları oy sayıları geldi,
df["vote_count"].describe([0.10,0.25,0.50,0.70,0.90,0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average",ascending=False).head(10)

from sklearn.preprocessing import  MinMaxScaler
#vote count'u 1 ile 10 arasında bastırdık
df["vote_count_score"] = MinMaxScaler(feature_range=(1,10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])

#vote average ile vote count score'u çarpıp yeni bir değişken elde ediyoruz
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)



#imdb weighted rating

#weighted rating = (v/(v+M) * r ) + (M/(v+M) * C)

#r = vote average
#v = vote count
#M = minimum votes required to be listed in the top 250
#c = The mean vote across the whole report (currently 7.0)

#1. film için 1i bölümün hesabı=(v/(v+M) * r ) =  örn film 1:   r = 8 , M = 500 , v = 1000
# (1000 / (1000+ 500))* 8 = 5.33
#1. film için 2. bölümün hesabı =  (M/(v+M) * C)
# 500/(1000+500)* 7 = 2.33
#toplam = 5.33 + 2.33 = 7.66


#oy sayısının etkisi matematiksel bir şekilde puana yansıtılmış oldu 1. bölüm
#örn film 2 :  r= 8, m = 500, v = 3000
#(3000 / (3000+500))*8 = 6.85
#2. bölüm
# 500/(3000+500)* 7 = 1
# toplam  = 7.85

#her iş yeri kendisine has bir sıralama yöntemi bulabilir.

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v/(v+M) * r ) + (M/(v+M) * C)

#weighted_rating(vote_average,vote_count, M,C) girerek her filmi hesaplıyabiliriz

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating",ascending=False).head(10)




#######################################
#Bayes average Rating Score

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1) * (n[k]+ 1) / (N + K)
        second_part += (k +1 ) * (k +1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df = pd.read_csv("week4/imdb_ratings.csv")
#hatalı yerleri almadık
df= df.iloc[0:, 1:]
#bar score feature'u oluşturup bar score'larını yazdırdık
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one","two","three","four","five",
                                                                "six","seven","eight","nine","ten"]]),axis=1)
#bar score'a göre ilk 10'u sıraladık
df.sort_values("bar_score", ascending=False).head(10)





