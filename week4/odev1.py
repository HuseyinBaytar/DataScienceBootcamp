
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
df = pd.read_csv("week4/datasets/amazon_review.csv")

df.head()

print(f"Ürünün Ortalama Puanı {df['overall'].mean():.3f}")


###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız. Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız
###################################################

df.loc[df["day_diff"] <= 30,"overall"].mean()
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()
df.loc[df["day_diff"] > 180,"overall"].mean()

#0-30 = 4.74  //   31-90 = 4.80  //  91-180 = 4.64 // 180 üzeri = 4.57

a = df.loc[df["day_diff"] <= 30,"overall"].mean() * 28/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 28/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24/100 + \
    df.loc[df["day_diff"] > 180,"overall"].mean() * 20/100

print(f"Zaman Aralıklı Ölçüm yaptığımızda; {a:.6f}")



###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################
###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
def score_up_down_diff(up,down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],x["helpful_no"]),axis=1)

#df["score_pos_neg_diff"] = score_up_down_diff(df["helpful_yes"],df["helpful_no"])

df.sort_values("score_pos_neg_diff", ascending=False).head(10)

score_up_down_diff(600,400) #%60
score_up_down_diff(5500,4500)  #%55

def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up+down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_average_rating", ascending=False).head(10)

score_average_rating(2,0)
score_average_rating(100,1)


def wilson_lower_bound(up,down,confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1- (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n )) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################
df.sort_values("wilson_lower_bound", ascending=False).head(20)
