################### what is machine learning?
#bilgisayarların insanlara benzer şekilde öğrenmesini sağlamak maksadıyla çeşitli algoritma ve tekniklerin geliştirilmesi için çalışılan bilimsel çalışma alanıdır.

#sayısal problemlere regresyon,  bağımlı değişkeni olanlar ise sınıflandırma problemidir.
# örneğin: ev fiyatları       ,  örneğin: titanic

#d##################eğişken türleri (variable types)
#sayısal değişkenler, yaş , metrekare, fiyat vs.

#kategorik değişkenler(nominal,ordinal): kadın erkek , hayatta kaldı kalamadı, futbol takımları
#nominal = futbol takımları gibi,  ordinal = eğitim durumu gibi

#bağımlı değişken ( target,dependent, output, response)
#örneğin bir hastalık tahmini durumunda, hasta olup olmadığı değişken bağımlı değişkendir.

#bağımsız değişken(feature, independent, input, column,predictor,explanatory)
#targeta etki ettiğini varsaydığımız diğer değişkenlerdir.


#################### Learning types
#makine öğrenmesi 3 farklı türde öğrenme türü olarak ele alınır
# Denetimli öğrenme(supervised Learning)
# Denetimsiz öğrenme( unsupervised Learning)
# Pekiştştirmeli öğrenme(reinforcement Learning)

#denetimli öğrenme: eğer veri setinde labellarımız(target) yer alıyorsa, denetimli öğrenmedir.
#bağımlı ve bağımsız değişkinler arasındaki ilişki öğrenilip, yeni gelen featureların targetini tahmin etmeye dayalıdır.

#denetimsiz öğrenme: Eğer veri setinde target yoksa, gözetimsiz denetimsiz öğrenmedir.
#clustering yapılabilir. segmentasyon vs.

#pekiştirmeli öğrenme : boş bir odada bir robot düşünün , o odadan çıkmaya çalıştığını düşün. yanlış her hareketinde cezalandırılsın ve pekiştire pekiştire öğrenip. önüne gelen engellere reaksiyonlar verip odadan çıkması.


###################Problem türleri.
#regresyon problemi mi?
#regresyon problemlerinde bağımlı değişken sayısaldır.

#Sınıflandırma problemi mi ?
#sınıflandırma problemlerinde bağımlı değişken kategoriktir.


##################model başarı değerlendirme yöntemleri

#tahminlerim ne kadar başarılı?
#regresyon modellerine başarı değerlendirme.

#diyelim ki 22.1 olan değere 20 dendi, 10.4 denen değere 11 dendi. kurmuş olduğumuz modeller tahminlerde bulunduğunda bu tahminlerde biraz sapmalar beklenir.

#mean squared error kullanarak başarı değerlendiririz. hemde optimizasyon yöntemlerinde bu fonksiyonlarda işimize yarar.
#yi = gerçek değerler, şapkalı y i = predicted değerlerr
#gerçek değerler ile preticted değerlerin kareleri alıp, toplamı alınmış yani 22.1 - 20 nin cevabı +  10.4 - 11'in cevabı ve diğer değişkenlerin  gerçek değerleri ile predictedlarını birbirinden çıkarıp hepsini toplayarak ilerleyip.  en son ortalamaları alınmış. ortalama hatayı almıştır.

#root mean squared error=  mse'den sonra karekök işlemi yapılır  geri dönüştürme metriği gibi düşünülebilir

#MAE = gerçek değerler ile tahmin edilen değerlerin farklarının mutlakları alınarak toplanır ve ortalaması alınır.
#mutlak almak = negatiflerinden kurtulmak

#sınıflandırma modellerinde başarı değerlendirme.

#başarılı yaptığı işler bölü bütün işler, yani doğru tahminleri, tüm işlemlere böleriz
#accuracy = doğru sınıflandırma sayısı  bölü toplam sınıflandırılan gözlem sayısı

#MSE ne kadar küçükse o kadar iyidir. Accuracy ne kadar büyükse o kadar iyidir.

#################### model validation yöntemleri

#elde ettiğimiz modellerin başarısını daha iyi elde etme çabasıdır.

#holdout yöntemi:
#orjinal veri setini eğitim seti olarak bölüp, birde test seti olarak bölünür.  eğitim setinde train/eğitim işlemi gerçekleşir. daha sonra test setine sorular sorulur. ve başarı bu şekilde elde edilir

#K-Katlı Çapraz Doğrulama (k fold cross validation)
#orjinal veri setinde 100 gözlem var diyelim, iki parçaya böldük %80'i eğitim %20si test, rastgele bölme işlemi gerçekleştirir.  şans eseri 20'lik kısımdaki gözlemlerin hepsi 1 gelirse, makine öğrenmesini iyi yapamıyacaktır. Her zaman geçerli değil tabi ki, eğer veri setinde çok fazla gözlem  varsa holdout gayet okeydir. Çapraz Doğrulama yöntemi 2 şekilde kullanılabilir. 1.si örneğin orjinal veri setini 5 parçaya böler, 4 parçayla model kurur. 1'iyle test et der, 5 parçanın hepsiyle sırasıyla bölüp test ettirebilir. 4'üyle eğitim 1'yle test şeklinde. bu yöntemle 5 farklı testin hata ortalmasını alıp Cross validation hata ortalaması elde edilir.
#2. yöntemde. holdouttdaki gibi 2ye bölünür eğitim %80, test %20 gibi, sonrasında %80lik kısma çapraz doğrulama yapılır, 4 parçayla model kur 1 parçayla test et gibi ilerler. En sonunda hiç görmediği yani ilk böldüğümüz %80/%20'deki %20'ye gider. %20 üzerinden performans testi edilir.
#elimizde çok fazla veri olduğunda en iyi olan, veriyi 2'ye ayırıp, train kısmında cross validation yapılmalıdır. en sondada test kısmında modeli test etmelidir.

###################Yanlılık-Varyans Değiş tokuşu (bias-VAriance tradeoff)

#underfitting : yüksek yanlılık
#doğru model : düşük yanlılık düşük varyans
#overfitting: yüksek varyans

#overfitting nedir?
#modelin veriyi çok iyi bir şekilde öğrenmesidir. Modelin veriyi değilde, verinin yapısını öğrenmesi gerekiyor. O yüzden model veriyi öğrenirse overfitting olur, veriyi ezberler.
#örn: bir çocuğa sınavından önce benzer sınav soruları verilir. çocuk o soruları ezberler ama sınavda farklı sorular çıkınca yapamaz. soruların yapısını, mantığını anlamaz sadece cevapları ezberler.

#underfitting nedir?
#varyans düşük ama yanlılık yüksek, bazı gözlemlere daha yakın ama diğerlerini pek umursamaz.

#aşırı öğrenme sorununu nasıl kaldırabiliriz ?
#eğitim hatasıyla test hatası kıyaslanır birbirinden ayrılmaya başladıkları noktada anlarız ki aşırı öğrenme gözlenir
#eğitim seti ve test seti içindeki hata değişimleri incelenir. eğitim seti ile test seti arasındaki ayrım başladığı yerde aşırı öğrenme tespit edilir. tam orada durdurulursa. aşırı öğrenme problemi çözülür.

#aşırı öğrenme nasıl çözülür?
#bir çok yanıtı var. veri setinin boyutu arttırabilir, feature selection yapılabilir. ama en nihayetinde test ile trainin ayrılmaya başladıkları yerden seçilip ayrım ypaılabilir.

#model karmaşıklığı : modelin hassaslaştırılması, daha detaylı tahminler yapabilmesi için özelliklerinin kuvvetlendirilmesi. mesela ağaç yöntemlerinde, bir ağaç 8 kere mi dallanacak yoksa daha fazla mı.
