#Dengesiz veri seti sınıflandırma problemlerinde görülür ve sınıf dağılımlarının birbirine yakın olmadığı durumlarda ortaya çıkar. Problem çoğunluğa sahip sınıfın azınlık sınıfını domine
#etmesinden kaynaklanır. Oluşturulan model çoğunluğa sahip sınıfa yakınlık gösterir, bu da azınlık sınıfının kötü sınıflandırılmasına sebep olur.

#Dengesiz veri setleriyle karşılaştığımızda doğru gözlem yapabilmek ve dengeyi sağlayabilmek için uygulayabileceğimiz çeşitli yöntemler vardır:

#Doğru Metrik Seçimi
#Precision-Recall-F1-score-ROC Curve-AUC
#Resampling
##Oversampling
#Random Oversampling - SMOTE Oversampling
##Undersampling
#Random Undersampling - NearMiss Undersampling - Undersampling (Tomek links) -Undersampling (Cluster Centroids)
#Daha fazla veri toplamak
#Sınıflandırma modellerinde bulunan “class_weight” parametresi kullanılarak azınlık ve çoğunluk sınıflarından eşit şekilde öğrenebilen model yaratılması,
#Tek bir modele değil , diğer modellerdeki performanslara da bakılması,
#Daha farklı bir yaklaşım uygulanıp Anomaly detection veya Change detection yapmak
#Dengesizlik içeren Credit Card Fraud Detection veri setini inceleyip, daha sonrasında bu dengesizlikle başa çıkabilmek için veri setine çeşitli yöntemler uygulayacağız.


#Accuracy sistemde doğru olarak yapılan tahminlerin tüm tahminlere oranıdır.
#Oluşturduğumuz modelin doğruluk skoru 0.999. Modelimiz mükemmel çalışıyor diyebiliriz, değil mi?
#Performansını incelemek için birde Confusion Matrix'ine bakalım.

#True Positives (TP) : Pozitif tahmin edildi ve bu doğru.
#True Negative (TN) : Negatif tahmin edildi ve bu doğru.
#False Positive (FP) : Pozitif tahmin edildi ve bu yanlış.
#False Negative (FN) : Negatif tahmin edildi ve bu yanlış.


#Resampling
#Yeniden örnekleme(Resampling), azınlık sınıfına yeni örnekler ekleyerek veya çoğunluk sınıfından örnekler çıkarılarak veri setinin daha dengeli hale getirilmesidir.

#Oversampling
#Azınlık sınıfına ait örneklerin kopyalanmasıyla veri setini dengeler.

#Random Oversampling:
#Azınlık sınıfından rastgele seçilen örneklerin eklenmesiyle veri setinin dengelenmesidir.
#Veri setiniz küçükse bu teknik kullanılabilinir.
#Overfitting’e neden olabilir.


#SMOTE Oversampling:
#Overfitting’i önlemek için azınlık sınıfından sentetik örnekler oluşturulması.
#Önce azınlık sınıfından rastgele bir örnek seçilir.
#Daha sonra bu örnek için en yakın komşulardan k tanesi bulunur.
#k en yakın komşulardan biri rastgele seçilir ve azınlık sınıfından rastgele seçilen örnekle birleştirilip özellik uzayında bir çizgi parçası oluşturarak sentetik örnek oluşturulur.


#Undersampling
#Çoğunluk sınıfına ait örneklerin çıkarılmasıyla veri setini dengeleme tekniğidir. Büyük veri setine sahip olunduğunda kullanılabilir. Elimizdeki veri seti büyük olmadığı için verimli sonuçlar
# alınmayacaktır. Ama yöntemleri açıklayıp bazılarının nasıl uygulanabiliceğini göstereceğim.

#Random Undersampling:

#Çıkarılan örnekler rastgele seçilir.
#Büyük veri setine sahipseniz bu tekniği kullanabilirsiniz.
#Rastgele seçimden dolayı bilgi kaybı yaşanabilir.


#NearMiss Undersampling:

#Bilgi kaybını önler.
#KNN algoritmasına dayanır.
#Çoğunluk sınıfına ait örneklerin azınlık sınıfına ait örneklerle olan uzaklığı hesaplanır.
#Belirtilen k değerine göre uzaklığı kısa olan örnekler korunur.

#Undersampling (Tomek links):

#Farklı sınıflara ait en yakın iki örneğin arasındaki çoğunluk sınıfının örnekleri kaldırılarak, iki sınıf arasındaki boşluk arttırılır.

#Undersampling (Cluster Centroids):

#Önemsiz örneklerin veri setinden çıkarılmasıdır.Örneğin önemli veya önemsiz olduğu kümelemeyle belirlenir.

#Undersampling ve Oversampling tekniklerinin bir araya gelmesiyle daha dengeli veri setleri oluşturulabilinir.



#iğer Yöntemler
#Daha fazla veri toplamak,
#Sınıflandırma modellerinde bulunan “class_weight” parametresi kullanılarak azınlık ve çoğunluk sınıflarından eşit şekilde öğrenebilen model yaratılması,
#Tek bir modele değil , diğer modellerdeki performanslara da bakılması,
#Daha farklı bir yaklaşım uygulanıp Anomaly detection veya Change detection yapmak
#gibi yöntemlerle de dengesiz veri setiyle başa çıkılır.

#Hangi yöntemin en iyi sonuç vereceği elimizdeki veri setine bağlıdır. Yöntemler denenerek veri setine en uygun olanın seçilmesi en iyi sonucu sağlar diyebiliriz.

