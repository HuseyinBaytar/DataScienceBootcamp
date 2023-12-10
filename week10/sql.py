## Structred Query Language kelimelerinin baş harflerinden oluşur SQL. Bütün veri tabanlarının ortak sorgulama dilidir. Veri tabanlarının ortak dilidir.
# SQL dili 1970'lerde ortaya çıkmış bir dildir. O zamanlardan bugüne hala popüler olarak kullanılır. Modern ingilizcede nasıl komut yazılıyorsa, konuşma diline en yakın şekilde bir sorgulama dili oluşturulmuştur.
# Veri bilimi projelerinin ham maddesi olan veriyi tedarik etmek için SQL dilini iyi bilmek gerekiyor. Veri ile ilgili açılan iş ilanlara bakıldığında SQL ve Pyhtonun öncelikli olduğunu görüyoruz.
# veritabanı temel anlamda verileri listeler halinde tablo ve satırlarda tutan her yapı aslında kendi çapında veritabanıdır. veri tabanları tablolar, sütunlar, satırlar ve indexlerden oluşur.
#veritabanı yönetim sistemleri, bizler bir veritabanından veriyi okurken bir takım kaynaklara ihtiyaç duyarız, cpu ram gibi kendi bilgisayarımızın kaynakları ham haldeki veriyi analiz ve sorgulamak için kullanılır. Oysa veri tabanı yönetim sisteminde bizler bir veritabanı sunucusuyla konuşuruz. Ondan bir veriyi getirmesini onun anlıyacağı bir dil ile söyleriz ve bu sistem bizim istediğimiz veriyi kendi kaynaklarını kullanarak bize iletir. Bu veri tabanı yönetim sistemi sadece bize değil bir çok istemciye cevap verir.
#veri tabanı sunucu, bir donanım değil bir yazılımdır. veri tabanı yönetim sisteminin diğer adıdır diyebiliriz. örneğin; İstemci bilgisayar bir veri tabanı sunucudan bir veriyi sorgulamak istiyor. bunun için öncelikle veri tabanı sunucusuna bağlanması gerekir, aynı ağda olmaları gerekir. Bu bağlantı sağlanınca sıradaki işlem gönderilen SQL dilinin çalıştırılmasıdır.  örneğin müşterilerin listesini çekmek için  SELECT * FROM Customer diyebiliriz.

#İlişkisel veritabanı Kavramı (RDMS) tekrar eden verileri tekilleştirmek amacı ile yapılandırılan veritabanı sistemleridir.  Tekrar eden verileri sürekli girmekten kaynaklanan emek ve iş güçü kaybını azaltmak için yapılmıştır. Tekrar eden verilerin gereksiz yere yer işgal etmesi ve kaynak israfını engellemek için yapılmıştır.
#Veri girerken insan hatası sebebiyle veri bütünlüğünün bozulmasını engellemek  ve veride geçmişe dönük olarak güncelleme yapmanın zor olması durumu yüzünden RDMS hayatımıza girdi.

#veri tipleri belirlediğimiz ve düzenlediğimiz işleme normalizasyon deriz.
#tam sayı veri tipleri

#big integer, -2^63 min ile maximum 2^63  8 byte yer kaplar
#integer , -2^31 ile 2^31 arasıdır  4 byte yer kaplar
#smallint , -2^15 ile 2^15 arasıdır, 2 byte yer kaplar
#tinyint , 0 ile 255 arasıdır, 1 byte yer kaplar
#bit 0 yada 1 değerlerini alır,

#string veri tipleri
#char = 0 ile 8000 arasında tanımlandığı değer kadar byte kaplar yanı char10 dersek 10 byte kaplar
#varchar = 0 ile 8000 arasında girilen değer uzunluğu + 2 byte yer kaplar
# varchar(max) 0 ile 2,147,483,647 arasındadır. kapladığı yer 2^31-1 bytedır
#text 0 ile 2,147,483,647  arasındadır. kapladığı yer 2,147,483,647
#ntext 0 ile 1,073,741,823 arasındadır. kapladığı yer 1,073,741,823

#ondalık veri tipleri
#decimal/numeric min -10^38+1 max 10^38-1  hassasiyetine göre diskte kapladığı alan değişir. 1 den 9 a kadar hassasiyet için 5 byte
#money= min -922,337,203,685,477.5808 , max pozitifi. 8 byte
#small money = min -214,748.3648 max tam tersi, 4 byte
#float 7 basamağa kadar 4 byte,  15 basamağa kadar 8 byte
#real, 4 byte

#tarih saat veri tipleri
#date  4 byte
#small date 3 byte
# datetime 8 byte, ayriyetten saatde ekleniyor
#datetime2, 6-7-8 byte arasında, milisaniyede ekleniyor
#datetimeoffset= 9-10 byte arasında, timezone aralığıda ekleniyor
#time sadece saat bilgisi tutulur.  5  byte default olarak kullanılırsa


#diğer veri tipleri
#image = artık sql server tarafından desteklenemeyen veri tipidir. resimden bağımsız, binary şekilde dosya saklamak için kullanılan veri tipi , 2 gb
#binary= tanımlandığı değere kadar byte kaplar,
#varbinary= 0 ile 8k arasında tanımlandığı değer +2 byte
#varbinary(max)= 2 gb'a kadar saklar
#xml= xml veriler için kullanılır
#table= sonradan kullanım amacıyla bir sonuç kümesini saklamak için kullanılır
#uniqueidentifier=  global olarak tekilliği garanti eden veriyi tutar
#hierarcyid, hiyerarşik yapılarda pozisyonları temsil etmek için
#geography = dünyadaki kordinat sistemini tutar
#geometry, euclidean sistemi ile kordinat sistemini tutar, dünyanın eğimlerini hesaba katmaz .

# SQL Dili
#2 parçaya ayırabiliriz.
#DML komutları (data manipulasyon)
# select, insert, update, delete, truncate
#datanın kendi üstünde işlem yapmak için kullandığımız komutlar

#SELECT: veritabanındaki tablolardan kayıtları çeker
#INSERT: tabloya yeni kayıt ekler
#UPDATE: tablodaki verilerin bir yada birden çok alanını değiştirir.
#DELETE: tablodan kayıt siler.
#TRUNCATE: Tablonun içini boşaltır.


#veri tabanı manipulasyon komutları
#create, alter, drop
#veritabanı nesnelerini oluşturmak, değiştirmek ve silmek için kullandığımız komutlar.
#CREATE: bir veritabanı nesnesini oluşturur.
#ALTER:  bir veri tabanı nesnesinin özelliğini değiştirir
#DROP:   bir veritabanı nesnesini siler.



#select komutu nasıl kullanılır?
# SELECT kolon1,kolon2,kolon3,....  FROM Tabloadı WHERE <şartlar>

#insert komutu nasıl kullanılır?
# INSERT INTO Tabloadı (kolon1,kolon2,...)  VALUES (değer1,değer2,....)

#update komutu nasıl kullanılır?
# UPDATE tabloadı SET kolon1=değer1,kolon2=değer2,.... WHERE <şartlar>

#datediff fonksiyonu iki yıl arasındaki farkı alır
#getdate = bugünün tarihini getiren fonksiyondur
#diyelim ki yaş eklemek istiyoruz UPDATE customers SET country="türkiye" , age= DATEDIFF ( year,birthdate, getdate())

#delete komutu nasıl kullanılır?
#DELETE FROM tablo WHERE <şartlar>

#WHERe komutu bir sql cümlesinde veriyi filtrelemek için kullandığımız bir sorgudur.
# SELECT kolon1,kolon2,... FROM tabloadı WHERE <şartlar>
#peki bu şartlar neler?
# = , <> , >, <, >= , <= , BETWEEN, LIKE , IN , NOT LIKE , NOT IN

#sql sorgusunda birden fazla koşul verebiliriz, AND ve OR operatörlerini kullanarak

#distinct komutu tekrar edenleri teke düşürür.

#ORDER BY komutu nasıl kullanılır? Sıralama komutudur.
#SELECT kolon1,kolon2.. FROM tabloadı  WHERE <şartlar>  ORDER BY kolon1 ASC, kolon2 DESC

#TOP komutu, bir sql sorgusundan belirli bir sayıdaki komutu almak için kullanılır
# SELECT TOP 10 kolon1,kolon2.. FROM tabloadı WHERE <şartlar>  ORDER BY kolon1 ASC, kolon2 DESC

############ AGGregate Functions ve Group By
# Veriyi çekerken tüm veriyi değilde, özet istediğimiz kısmı çıkarmak için kullanılan sorgulardır.
#aggregate functions (min,max,sum,avg,count...)


#SELECT aggfunction adı(SUM,COUNT.etc) FROM tabloadı WHERE city=istanbul


#primary key= her bir tabloyu tekil olarak tanımlayan o tablonun unique değeri, örneğin id
#foreign key = başka tablolarda referans içeren bir değer var ise foreign key diyoruz.

#- Örnek bir E-Ticaret Sistemi;
#1. Kullanıcı sisteme login olur - Adres/Telefon/Email gibi bilgileri sistemde kayıtlıdır.
#2. Seçtiği ürün ya da ürünleri sepete ekler. - Adres listesinden adres seçer.
#3. Sepete eklediği ürünleri ödeme ekranına girer - Bu sırada sipariş oluşturulur / Kredi kartı ödemesi gerçekleşir.
#4. Ürün sevkedilir. - Faturası kesilir ve ürün sevkedilir.

#- JOIN türleri;
#diyelim ki USERS ve ADDRESS tabloları var elimizde, kullanıcısı olup adresi olmayan veya adresi olup kullanıcısı olmayan kayıtlar olabilir,
# ayrıca bazı kullanıcıların 1'den fazla adresi de olabilir böyle durumlarda;
#Inner Join ile her iki kümenin kesişimini alırız.
#Diğer türü; Left (Outer) Join, her iki tarafta olan satırlar dolacak hem users hem address, ek olarak tek bir tabloda olanlarda gelecek ama
#usersta var address te bilgisi yoksa address satırı boş gözükecek.
#Right (Outer) Join, kesişimler yine gelecek fakat bu sefer sadece sağ tarafta bilgisi olan yani address satırı gelecek ama users satırı boş duracak o satır için..
#Full (Outer) Join; kesişimler + tek bilgisi olanlarda gelecek
#Kurumsal işlerde left join özellikle kullanılması öneriliyor.


#- Subquery , ilişkisel veri tabanlarında farklı bir birleştirme yöntemidir. Bir sorgunun içerisinde başka bir sorguyu bir kolon gibi kullandığımız yapıya subquery deniyor.
# Fakat join kullanmaya göre daha az performanslıdır , daha çok yer tutar.




