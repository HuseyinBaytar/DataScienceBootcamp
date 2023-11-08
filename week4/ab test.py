                # A/B test

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.options.display.width = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#a bir özelliği temsil etsin, b farklı bir özelliği temsil etsin, bu ikisi arasında farklılık olup olmadığını ilgilendiğimiz konudur

#############################################
                # sampling(örnekleme)
#bir ana kitle içerisinden, bu ana kitleninin özelliklerini iyi taşıdığı varsayılan bir alt kümedir.


populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()
#ortalama 39.4, hepsinin yaşı olsa ortalamayı hesaplamak için 10 bin kişiye gidip tek tek sorulması lazım ama örnek theorysi der ki, sen o 10 bin kişiyi temsil eden iyi bir alt küme seç bu rastgele ve yansız olsun, o ana kitleyi iyi temsil etsin, 10 bin kişiyi gezmeden bir genelleme yapma şansı verir.
#seed verdik bir sonraki denemelerde aynı olsun diye
np.random.seed(115)
#random  100 kişi çektik
orneklem = np.random.choice(a=populasyon, size= 100)
#100 kişinin meanine bakıyoruz 39.06 neredeyse 10bin kişi ile aynı.
orneklem.mean()

#seed verdik
np.random.seed(10)

#10 tane farklı örneklem yaptım ve hepsinin ortalamasını alıcam
orneklem1 = np.random.choice(a=populasyon, size= 100)
orneklem2 = np.random.choice(a=populasyon, size= 100)
orneklem3 = np.random.choice(a=populasyon, size= 100)
orneklem4 = np.random.choice(a=populasyon, size= 100)
orneklem5 = np.random.choice(a=populasyon, size= 100)
orneklem6 = np.random.choice(a=populasyon, size= 100)
orneklem7 = np.random.choice(a=populasyon, size= 100)
orneklem8 = np.random.choice(a=populasyon, size= 100)
orneklem9 = np.random.choice(a=populasyon, size= 100)
orneklem10 = np.random.choice(a=populasyon, size= 100)

# 10 örneklemin ortalaması 40.08, biraz daha yaklaştık ortalamaya
(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

#burda anlatılmak istenen, daha az veri ile belirli bir yanılma payı ile  aynı sonucu alabiliceğimiz.



############################################
# Descriptive Statistics ( Betimsel İstatistikler)

#elimizdeki veri setini betimlemeye çalışma çabasıdır

df = sns.load_dataset("tips")
#count,mean,std,min,çeyrek değerler, max verir.
df.describe().T
#çeyrek değerler bize bilgi vermektedir, mean ile %50 birbirine yakın olması vs



#####################################################
#confidence Intervals (Güven aralıkları)

#Anakitle parametresinin tahmini değerini kapsayabilecek iki sayıdan oluşan bir aralık bulunmasıdır.

#örnek, web sitesinde geçirilen ortalama sürenin güven aralığı nedir?
#ortalama 180 saniye, standart sapma 40 saniye
#kullanıcıların websitede geçirdiği ortalama %95 güven ile 172 ile 188 saniye arasıdır.
#100 kullanıcıdan 95'i ortalama 172-188

#adım1: n(örneklem),ortalama ve standart sapmayı bul
#adım2: güven aralığına karar ver 95 mi 99 mu
#adım3: değerleri kullanarak güven aralığnı hesapla

df.describe().T
#ortalaması 19.7

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
#18.6 ile 20.9  güven aralığı verdi

sms.DescrStatsW(df["tip"]).tconfint_mean()
#bana gelen müşterilerimin bırakacak oldukları bahşişler 2.8 ile 3.1 aralığında olacaktır.

#elimdeki sayısal değiklenlerin ortalamasına erişkin tek bir değer yerine, olası çekilebiliecek başka ortalamalar söz konusu olsaydı bu ortalamaların ne şekilde olabiliceği ile ilgili daha güvenli bir bilgiye sahip oluyorum.


######################################################
#Correlation

#Değişkenler arasındaki ilişki, bu ilişkinin yönü ve şiddeti ile ilgili bilgiler sağlayan istatiksel bir yöntemdir.

df["tip"].corr(df["total_bill"])
#ödenen hesap miktarı arttıkça bahşişte artar o yüzden pozitif korelasyon vardır


#######################################################
# Hypothesis Testing (hipotez testleri)

#bir inanışı, bir savı test etmek için kullanılan istatistiksel yöntemlerdir.

#Grup karşılaştırmalarında temel amaç olası farklılıkların şans eseri ortaya çıkıp çıkmadığını göstermeye çalışmaktır.
#iki grubu kıyaslarken farkı görüyor olucaz ama bu farkın şans eseri çıkıp çıkmadığını göstermeye çalışcaz


#örn: Mobil uygulamada yapılan arayüz değişikliği sonrasında kullanıcıların uygulamada geçirdikleri günlük ortalama süre arttı mı ?

#arayüz değişikliği öncesi : A , arayüz değişikliği sonrası : B olarak modelliyoruz
# kullanıcıların uygulamada geçirdiği sürede fark yoktur diye bir hipotez kuruyoruz daha sonra bunu test ediyoruz

#diyelimki bu iki değişiklik neticesinde, kullanıcıların 1 kısmına A, diğer kısmına B gösterilsin ve ölçüm yapılsın.
#1. tasarımda 55 dakika
#2. tasarımda 58 dakika

#ama burda 58 dakika oldu diye, 2. tasarım daha iyi denilebilir mi?  Hayır denmez, A/B testlerinin en kritik noktası burasıdır.
# matematiksel olarak 58 daha iyi gözükebilir ama biz örnek aldık, bu farklılık şans eseri ortaya çıkmış olabilir.
#buradaki farkın şansa yer bırakmayacak şekilde ortaya çıkıp çıkmadığını, istatistiki bir şekilde ispat etmemiz gerekir.



##################################################################
# AB Testi (bağımsız iki örneklem T testi)

#2 grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# H0 :  iki grup ortalaması birbirine eşittir,  küçük/eşittir   ya da  büyük/eşittir olarak 3'e ayrılır
# H1 :  iki grup ortalaması birbirine eşit değildir, büyüktür ya da küçüktür olarak 3'e ayrılır.

# p value değerine bakarak yorumluyor olucaz, p value eğer 0.05 den küçükse, H0 red diyecez.
# bağımsız iki örneklem t testinin 2 varsayımı vardır, normallik ve varyans homojenliği,  yani normal dağılım ve 2 grubun birbirine benzer olması


# 1. adım: Hipotezleri Kur.
# 2. adım: Varsayım Kontrolü ( normallik varsayımı / Varyans homojenliği)
# 3. adım: Hipotezin Uygulanması
#  * varsayımlar sağlanıyosa bağımsız iki örneklem T testi yani A/B testing
#  * varsayımlar sağlanmıysa mannwhitneyu testi
# 4. p- value değerine göre sonuçları yorumla
# not:
# - normallik sağlamıyorsa direkt 2 numara. varyans homojenliği sağlanmıyorsa 1 numara argüman girilir.
# - normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir


#############################################################################
# uygulama 1 : sigara içenler ile içmeyenlerin hesap ortalamaları arasında ist ol an fark var mı?
#############################################################################

df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})
#fark var gibi gözüküyor ama istatiksel olarak far mı ?, bu fark şans eseri olarak mı ortaya çıktı?

# 1- hipotezi kur.
# H0 : M1=M2
# H1 : M1 != M2

# 2- Varsayım kontrolü

#####################normallik varsayımı

# H0: normal dağılım varsayımı sağlanmaktadır
# H1: ... sağlanmamaktadır.

#shapiro testi ile normallik varsayımını kontorl ediyoruz
test_stat, pvalue = shapiro(df.loc[df["smoker"]== "Yes", "total_bill"])
print('test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p value değeri 0.05den küüçük olduğu için H0 Red, normal dağılım varsayımı sağlanmamaktadıra çıkar.

test_stat, pvalue = shapiro(df.loc[df["smoker"]== "No", "total_bill"])
print('test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# 0.00 oldğu için  h0 red, normal dağılım varsayımı  sağlanmamaktadır.
#nonparametrik kullanmamız lazım .


##################### varyans homojenliği varsayımı
# H0: varyanslar homojendir
# H1: varyanslar homojen değildir.

# varyans homojenliği testi
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"]== "No", "total_bill"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))
#p value < 0.05den H0 Red / vayanslar homojen değilmiş.

#####################
# 3. hipotezin uygulanması
#####################

#  * varsayımlar sağlanıyosa bağımsız iki örneklem T testi yani A/B testing (sadece görmek için deniyoruz
#eğer normal dağılım ve varyans kabul ediliyo olsaydı equal_var = True, fakat varyans sağlanmayınca equal_var=False olarak kullanılacak
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"]== "No", "total_bill"],
                              equal_var=True)
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

# sigara içenler ile içmeyenler arasında istatistiksel olarak  fark yoktur.


#  * varsayımlar sağlanmıyosa mannwhitneyu testi    (bizim senaryomuzda sağlanmıyor)

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"]== "No", "total_bill"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#ikisinin ortalamaları arasında anlamlı bir farklılık yoktur. H0 red edilemez.

#########################################################
#uygulama 2 : titanic kadın ve erkek yolcuların yaş ortalamaları arasında istatistiksel olarak anlamlı bir fark var mı?
############################################################

df = sns.load_dataset('titanic')
df.head()

df.groupby("sex").agg({"age": "mean"})
#fark var gibi ama şans eseri mi ?


#1. hipotezleri kur:
# h0: m1 = m2 (yaş ortalamaları arasında anlamlı bir fark yoktur)
# h1: m1 != m2 (yaş ortalamaları arasında anlamlı bir fark vardır)

#2. varsayımmları incele

#normallik varsayımı
#hO: normal dağılım varsayımı sağlanmaktadır
#h1: ... sağlanmamaktadır

test_stat , pvalue =shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))
#h0: red edilir,  normal dağılım sağlanmıyor.

test_stat , pvalue =shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))
# h0: red edilir, normal dağılım sağlanmıyor pvalue 0.05 den küçük


#varyans homojenliği
# h0: varyans homojen
# h1: varyans homojen değil

test_stat , pvalue =levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#varyanslar homojendir h0'ı red edemeyiz pvalue 0.05den yüksek fakat   2 varsayım sağlanmadığı için nonparametrik kullanıcaz


test_stat , pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#h1: kadın ve erkek yolcuların arasında gözlemlediğimiz fark, istatistiksel olarakta vardır.





#############################################################
# uygulama 3: Diyabet hastası olan ve olmayanların yaşları arasında ist. ol. anl. fark var mıdır?
#############################################################

df = pd.read_csv("week4/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})
#1= diyebet, 0= not diabet
# 1'in ortalaması 37, 0'ın ortalaması 31, şans eseri mi yoksa istatiksel mi?

# 1. adım: Hipotezleri Kur.
#h0: m1 =m2 = fark yoktur
#h1: m1 != m2 = fark vardır

# 2. adım: Varsayım Kontrolü ( normallik varsayımı / Varyans homojenliği)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))
#normallik var sayımı sağlanmamaktadır = mannwithyu'ya gitmemiz lazım

# 3. adım: Hipotezin Uygulanması
#  * varsayımlar sağlanıyosa bağımsız iki örneklem T testi yani A/B testing
#  * varsayımlar sağlanmıysa mannwhitneyu testi
test_stat , pvalue = mannwhitneyu(df.loc[df["Outcome"] == 0, "Age"].dropna(),
                           df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#h0 red edilir, iki ortalama birbirine eşittir. istatistiksel olarak anlamlı bir fark vardır.

###############################################################################################
#iş problemi: Kursun büyük çoğunluğ*unu izliyenler ile izlemiyenlerin puanları birbirinden farklı mı?
########################################################################################
#H0 : M1 = M2 ( fark yoktur)
#h1: M1 != M2 ( fark vardır)

df = pd.read_csv("week4/datasets/course_reviews.csv")
df.head()
#çoğunu izliyenler 4.86
df[(df["Progress"]> 75)]["Rating"].mean()
#azını izliyenler 4.70
df[(df["Progress"] < 10)]["Rating"].mean()


test_stat, pvalue = shapiro(df[df["Progress"] > 75]["Rating"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[df["Progress"] < 25]["Rating"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))



test_stat, pvalue = mannwhitneyu(df[df["Progress"] > 75]["Rating"],
                                 df[df["Progress"] < 25]["Rating"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))


# kursu daha fazla izliyenlerin, daha az izliyenlere göre verdikleri puan ortalamları daha yüksek, h0 red edilir


##################################################################################################
# iki örneklem oran testi,  iki grup oran karşılaştırma
##########################################################
#önceden elimizde ortalama vardı şimdi oranlar olacak
# h0 : p1 = p2
# h1 : p1 != p2


# h0: yeni tasarımın dönüşüm oranı ile eski tasarımın oranı arasında fark yoktur
# h1: ... vardır.

basari_sayisi = np.array([300,250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
# p değeri 0.05 den küçük olduğu için h0 red ederiz, yani iki oran arasında anlamlı bir farklılık vardır.




####### uygulama: kadın  ve erkeklerin hayatta kalma oranları arasında anlamlı bi fark var mı?
#h0: p1 = p2
#fark yoktur
#h1: p1 != p2
#fark vardır

df = sns.load_dataset("titanic")
df.head()

#bariz bir fark var ama pratik olsun diye yapıyoruz
df.loc[df["sex"] == "female", "survived"].mean()
df.loc[df["sex"] == "male", ("survived")].mean()


female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

# h0 red edilir, fark vardır.



###################################################################
                #ikiden fazla grup ortalama karşılaştırma
#(ANOVA - Analysis of Variance)

#hipotezimiz = h0: p1=p2=p3 gruplar birbirine eşittir
# h1: p1!=p2!=p3  eşit değildir olarak kurulmuştur

df= sns.load_dataset("tips")
df.head()

#haftanın günleri açısından, ödenen hesaplar ile ilgili bir farklılık var mı ?
#hafta içi birbirine yakın, haftasonları birbirine yakın ama sanki haftaiçi ile haftasonun arasında fark var gibi, şans eseri mi ?
df.groupby("day")["total_bill"].mean()

# h0: m1= m2= m3= m4
# h1: fark vardır

#varsayım kontrolü
#eğer varsayım varsa one way anova, varsayım sağlanmıyorsa kruskal
# h0: normal dağılım varsayımı sağlanmaktadır

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)

#hepsi 0.05 den küçük olduğu için h0 red edilir, normallik varsayımı sağlanmamaktadır.

#h0: varyans homejenliği varsayımı sağlanmaktadır.
test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#pvalue 0.5, h0 red edilemez,  homajenlik varsayımı sağlanmaktadır ama , normal dağılım varsayımını geçemediği için her türlü nonparametrik gidiecez.


#3. hipotez testi ve p -value yorumu

#parametrik anova testi: eğer varsayım sağlansaydı, ki bu senaryoda sağlanmıyor.

f_oneway(df.loc[df["day"] == "Sun", "total_bill"],
                  df.loc[df["day"] == "Sat", "total_bill"],
                   df.loc[df["day"] == "Thur", "total_bill"],
                   df.loc[df["day"] == "Fri", "total_bill"])
#p value 0.5 den küçüktür, h0 red edilir. gruplar arasında istatiksel oalrak anlamlı bir fark vardır.

#nonparametrik test: bu senaryo için geçerli olan
kruskal(df.loc[df["day"] == "Sun", "total_bill"],
                  df.loc[df["day"] == "Sat", "total_bill"],
                   df.loc[df["day"] == "Thur", "total_bill"],
                   df.loc[df["day"] == "Fri", "total_bill"])
#bu testin sonucunda p value 0.05 den küçüktür, h0 red edilir, istatistiksel olarak anlamlı bir fark vardır.





#2 li karşılaştırmalar yaparak gördük hangilerinin değerleri nasıl diye
from statsmodels.stats.multicomp import  MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())















