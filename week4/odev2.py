#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.options.display.width = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control = pd.read_excel("week4/datasets/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("week4/datasets/ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
control.head()
control.describe().T

test.head()
test.describe().T
# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat([control,test])
df.describe().T



#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 (Control ve Test Grubundakiler Arasında Purchase açısından İstatistiksel  Bir Fark Yoktur)
# H1 : M1!= M2 (Control ve Test Grubundakiler Arasında Purchase açısından İstatistiksel  Bir Fark Vardır)

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
control['Purchase'].mean()
test['Purchase'].mean()


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
test_stat, pvalue = shapiro(control['Purchase'])
print('test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#control grubunun normalliği red edilemez

test_stat, pvalue = shapiro(test['Purchase'])
print('test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#test grubunun normalliği red edilemez.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

test_stat, pvalue = levene(control['Purchase'],
                           test['Purchase'])
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))

#varyans homojenliği red edilemez.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.


#varsayımlar sağlandığı için ve veri normal olduğu için, parametrik hesaplama yöntemi seçmemiz lazım, A/B testi yapılabilir.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
test_stat, pvalue = ttest_ind(control['Purchase'],
                              test['Purchase'],
                              equal_var=True)
print('Test stat = %.4f , p-value = %.4f' % (test_stat, pvalue))


#pvalue 0.34 olduğu için, h0 kabul edilir, Control ve Test Grubundakiler Arasında İstetistiksel Açıdan Bir Fark Yoktur,
# normallik dağılımı ve varyans dağılımı  doğru olduğu için, parametrik hesaplama yöntemi olan ttest_ind yani,
#bağımsız örneklem T testi (a/b testi) yaptım.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Burada control grubu ile test grubu arasında anlamlı bir farklılık bulunmadığından değişime gidilmesi tercihe bırakılmalıdır.
# Burada tıklanma sayısı gibi diğer ölçütlere de bakılabilir veya daha fazla gözlem yapılıp  o şekilde karar verilmelidir.











