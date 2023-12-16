# Time Series Forecasting

#Zaman serisi nedir?
#zamana göre sıralanmış gözlem değerlerinden oluşan veridir. Örn; borsa, hava durumu verileri vs.
#Stationary, Trend, Seasonality, cycle bu kavramlar zaman serisi yorumlamakta çok önemlidir.

#Stationary
#bir serinin istatistiksel özelliklerinin zaman içerisinde değişmemesine denir. Bir zaman serisinin ortalaması, varyansı ve kovaryansı zama nboyunca sabit kalıyorsa, serinin stationary olduğu
#ifade edilir. Teorik olarak zaman serisi verilerinin yapısı belirli bir patern ile ilerliyorsa, daha tahminedilebilir olur. Eğer zaman serisi stationaryse daha rahat tahminde bulunabilir.
#Eğer bir seride stationary olmama durumu gözlendiyse serinin farkı alınır. T zamanında pazartesi değerleri var diyelim, bide pazardaki değerleri var, bu iki değer birbirinden çıkarılır. seri
#durağan hale getirilir.

#Trend
# Zaman serisi konusu için en kritik başlıklardan biridir. Her alanda Trend kavramı farkındalığı taşınması gerekir. Bir zaman serisinin uzun vadedeki artış ya da azalışının gösterdiği yapıya
#trend denir. Trend var ise Stationary olma ihtimali çok düşüktür.  

#Seasonality
#zaman serisinin belirli bir davranışı belirli periyotlarla tekrar etmesi durumuna mevsimsellik denir.

# Döngü(Cycle)
# Döngüsellik mevsimselliğe benzer bir yapıdır. Fakat ikisi farklıdır, forecasting açısından çok kritik bir önem taşımıyor olsada bilmek gereklidir. Mevsimsellik daha belirgin, kısavadeli, düzenli aralıklarla
#gün hafta meysim gibi şeylerle örtüşecek şekilde ilişkilendirilir.
#Döngüsellik ise daha uzun vaadeli daha belirsiz bir yapıda, gün hafta mevsim gibi şeylerle örtüşmeyecek şekilde ilişkilendirilir. daha çok yapısal nedenlerle ortaya çıkar, örneğin politika dünyasındaki
#bazı kişilerin açıklamalarıyla değişir.

###### Zaman serisi Modellerinin doğasını anlama
# Bir sonraki günün olacaklarını tahmin etmek, bizim amacımız. bir zaman serisi periodundaki gelecek gün en çok kendinden bir gün önceki değerden etkilenerek ilerler. Bu varsayımdan hareketle
#kendisinden önceki 4-5 değerin ortalamasını alıp bir sonraki değerin tahmini olarak verebiliriz. Ama diğer yandan dedik ki bu seri önceden çok alçalıp yükseldi, bu noktaların aynı tarihteki ortalamlarını
#alıp bunuda üzerine ekliyelim diyebiliriz. Geçmişten gelen bir mevsimsellik bilgisini taşımak ve geçmiş değerlere odaklanmak mantıklı.

### Moving Average
#Bir zaman serisinin gelecek değeri kendisinin k adet önceki değerinin ortalamasıdır. Hareketli ortalama genelde tahmin etme için değilde, bir trendi yakalamak ve gözlemlemek için kullanılır.
#fakat ML kapsamında feature'lar türetirken, haraketli ortalamaya dayalı özellikler türetiyor oluruz.

### Weighted Average
#Ağırlıklı ortalama, hareketli ortalamaya benzer. Daha sonlarda olan gözlemlere daha fazla ağırlık verme fikrini taşır. Yani diyelim ki weighted average alırken, ilk 4 güne daha çok ağırlık sonrasındaki günlere gittikçe azalan şekilde ağırlık vererek dengeli bir tahmin yapabiliriz.

#Anlaşıldığına göre  zaman serileri verileri, kendisinden önceki değerlerden daha fazla etkileniyor. Dolayısıyla kendisinden önceki değerlere gitme işi bizim için bir odak, Kendisinden önceki verilere nasıl gidileceğide bir odaktır, örneğin moving average veya Weighted average.

