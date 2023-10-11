#numpy = numerical pyhton'un kısaltılmışıdır
#matematik ve istatistik alanında yapılmış bir kütüphane
#listelere kıyasla, numpy sabit bir type ile tutar ve işlem yapma olanağı sağlar
#daha az çabayla, daha fazla işlem yapma olanağı sağlar

#numpy kütüphanesini 'np' olarak çagiriyoruz
import numpy as np
a = [1,2,3,4]
b = [2,3,4,5]     #2 tane liste oluşturduk

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

#yukarıda  gördüğümüz fonksiyon ile ilgili indexleri yakalayıp birbiriyle çarpıp, yeni bir listeye atması için
#yazılan bir fonksiyondur, pyhtonic way olarak böyle yazılabilir ama numpy ile daha kısa bir şekilde yapılabilir.
# listeyi numpy arraye çevirip, a çarpi b şeklinde yazdırabiliriz
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a * b

#neden numpy?
#1- hız, sabit veri tutar bu yüzden hızlıdır
#2- yüksek seviyeden işlemler yapmaya kolayca olanak sağlar.

#########   NumPy Array'i oluşturmak   #########

np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

#numpy kütüphanesinden zeros methodunu çağırdık, bu method içine girilen parametre kadar sayı oluşturur.
np.zeros(10, dtype= int)
#random sayı oluşturmak için ypaılan method
np.random.randint(0, 10, size= 10)      #belirli bir sayı aralığında random sayı seçtirdik

np.random.normal(10,4,(3,4))  #ortalaması 10, standart sapması 4, 3e4'lük bir random sayı oluşturduk


####### Numpy Array Özellikleri ################
#öncelikle kendimize bir np arrayi oluşturalım
a = np.random.randint(10, size=5)

a.ndim   #boyut sayısı / 1 boyulu olduğu için 1 gelir
a.shape  #boyut bilgisi / tek boyutlu ve içinde 5 tane eleman var demiş olur
a.size   #toplam eleman sayısı
a.dtype  #array veri tipi

#######  Yeniden şekillendirme ###########
#tek boyutlu bir matris vardır elimizde
np.random.randint(0, 10, size= 9)
#3e 3'lük bir boyuta çevirmek için  .reshape(3,3) kullandık  # 9 elemanlı olduğu için 3-3'e çevirdik, 10 olsaydı hata verirdi
np.random.randint(0, 10, size= 9).reshape(3,3)


########## Index Seçimi ######################
#random 10'a kadar, 10 elemanlı değer oluşturduk
a = np.random.randint(10, size= 10)
#atanan değer [liste içinden verilen numara] ile istediğimiz değere gidebiliriz, burda 1. elemanına gittik
a[0]       #bu işleme index seçimi
#0:5 dediğimizdede 0 dahil, 5 e kadar gidiş yaptık yani 0-1-2-3-4'üncü elemanlarını alır
a[0:5]     #bu işlemede slicing denir, iki nokta üstüste ile bir yere kadar belirtiriz
a[0] = 999   # bu atama ile 1. değerimizi 999'a çeviriyoruz

#0dan 10'a kadar random sayılardan, 3 e 5'lik bir değer oluşturduk
m = np.random.randint(10, size=(3,5))
#ilk ifadesine erişmek istersek eğer,   virgülden öncesi satırları, virgülden sonrası sütunları temsil eder.
m[0,0]
m[1,1]  #1e1 gittik
m[2,3]  #2. satır, 3. sütuna

m[2,3] = 999 #999'a atadık

#numpy fix type arraydir, biz bu listemizi önceden integer atadığımız için, float atarsak içine, integer olarak girecektir
m[2,3] = 2.9

# : ile bütün satırları seç dedik, virgül ilede verilen sütünü seçtirdik
m[:,0]
# 1. satırı ve tüm sütunları seçmek içinde aşşağıdaki ifadeyi yazarız
m[1, : ]
# satılarda 0dan2'ye kadar git, sütünlardada 0 dan 3'e kadar git dedik
m[0:2, 0:3]

########### Fancy Index #############
#arrange ifadesine aşşağıda 0 dan 30'a kadar 3'er 3'er artacak şekilde değeri girdik
v= np.arange(0, 30, 3)
#4. indexi seçtik
v[4]

catch = [1,2,3]   #1-2-3 değerleri olan bir liste oluşturduk
#listeyi numpy arrayine soktuğumuzda, v'nin içindeki 1-2-3 değerlerini döndürür, listede o değerler olduğu için
v[catch]


############  Numpy'da koşullu işlemler ###############

v=np.array([1,2,3,4,5])

#amacımız 3'den küçük olanlara erişmek
v < 3  #3'den küçük olanları sorguladık ve bize true false döndürdü

v[v < 3]  #bu komutu yollayınca içeir, true olarak gördüklerini seçip, false olarak gördüklerini seçmez
v[v > 3]  #3'den büyükler
v[v != 3] #3'hariç hepsi
v[v == 3] #sadece 3
v[v >= 3] #3 ve üzeri

################ matematiksel işlemler #####################
#1-5 arası bir array oluşturup v'ye atadık
v = np.array([1,2,3,4,5])
#v /5 dediğimizde bütün elemanlarını 5'e bölüp ekranda döndürür
v / 5
# v'yi 5 ile çarpıp 10'a böldük
v *5 /10
#v'nin karesini alırız
v **2
#v'den çıkarma işlemi
v -1

#np.subtract = çıkartma işlemi
#np.add = toplama işlemi
#np.mean = ortalamasını alır
#np.sum = toplamını alır
#np.min = minimum değerini getirdi
#np.max = maximum değerini getirdi
#np.var = varyansını getirir


# numpy ile iki bilinmeyenli denklem çözümü

# 5*x0 + x1 = 12
# x0 + 30x1 = 10

a = np.array([[5,1], [1,3]])
b = np.array([12,10])
#bilinmeyen değerleri hesaplayan bir fonksiyon
np.linalg.solve(a,b)


