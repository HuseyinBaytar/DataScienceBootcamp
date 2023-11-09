#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################
#diyelim ki bir kişinin izlediği filmler ile diğer kişilerin izlediği filmler benzer ama bir kullanıcı 3 filme oy vermedi, diğerleri verdi, o oy vermediği boşlukları doldurmak için kullanılan bir modelleme tekniğidir.

#boşlukları doldurmak için user'lar ve movie'ler için var olduğu varsayılan latent featurelarının(gizli faktörler)  ağırlıkları var olan veri üzerinden bulunur ve bu ağırlıklar ile var olmayan gözlemler için tahmin yapılır.

# user-item matrisinin 2 tane daha az boyutlu matrise ayrıştırılır.
#2 matristen user-item matrisine gidişin latent factorler ile gerçekleştiğini varsayımında bulunur.
#dolu olan gözlemler üzerinden gizli faktörlerin ağırlıkları(latent factor) bulunur
#bulunan ağırlıklar ile boş olanlar doldurulur


#rating matrisinin iki factor matrisin çarpımı(dot product) ile oluşturulur
#factor matrisler? user latent factor, motive latent factor
#latent factor? latenent fature? gizli faktörler yada değişkenler
#kullanıcıların ve filmlerin latent feature'lar için skorelara sahi polduğu düşünülür.
#bu ağırlıklar önce var olan veir üzerinden bulunu ve sonra boş bölümler bu ağırlıklara göre doldurulur.

#mesela fimlerin türleri gizli faktör olabilir, veya belirli bir oyuncunun olması.

#var olan değerler üzerinden iteratif şekilde tüm p ve q'lar bulunu rve sonra kullanılır.
#başlangıçta rastgele p ve q değerleri ile rating matrisindeki değerler tahmin edilmeye çalışır.
#her iterasyonda hatalı tahminler düzenlenerek rating matristeki değerlere yaklaşılmaya çalışılır.
# örneğin bir iterasyonda 5'e 3 dendiyse sonrakinde 4 sonrakinde 5 denir.
#böylece belirli bir iterasyon sonucunda p ve q matrisleri doldurulmuş olur.

####Gradyan descent
#fonksiyon minimizasyonu için kullanılan bir optimizasyon yöntemidir
#negatif olarak tanımlanan en dik iniş yönünde iteratif olarak parametre değerlerini güncelleyerek ilgili fonskiyonun minimum değerlerini verecek parametreleri bulur.


import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('week5/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('week5/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

#4 tane film seçiyoruz örnek olarak
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

#veri setini indirgedik
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

#97 bin adet rate var
sample_df.shape

#yeni df oluşturuyoruz
user_movie_df = sample_df.pivot_table(index=["userId"],columns=["title"],values="rating")
#satırlar rateleri, sütunlar filmleri
user_movie_df.shape

#ratelerin scalası olarak 1 ile 5 arasında bir skala veriyoruz
reader = Reader(rating_scale=(1, 5))

#surprise için uygun bir veriye çeviriyoruz
data = Dataset.load_from_df(sample_df[['userId','movieId','rating']], reader)

##############################
# Adım 2: Modelleme
##############################
#oluşturduğumuz veri setinden train test seti oluştururuz, trainde antreman yapar, test setinde test eder
#2'ye ayırıyoruz
trainset, testset = train_test_split(data, test_size=.25)

#modeli kuruyoruz
svd_model = SVD()
#modeli fit ediyoruz
svd_model.fit(trainset)
#test setinin değerlerini tahmin ediyoruz
predictions = svd_model.test(testset)

#hata ortalaması
accuracy.rmse(predictions)

#bladerunner filminin tahminlerine baktık 1 numaralı kişinin
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################
#kullandığımız modeli optimize etmek için yaptığımız bölüm,  tahmin performansını arttırmaya çalışıyoruz.

#n factor parametresini verirsek, dışardaki faktorlere bakar deafultu 20
param_grid = {'n_epochs': [5, 10, 20],'lr_all': [0.002, 0.005, 0.007]}

#modeli veriyoruz, parametre çiftlerini veriyoruz, rmse ve mae'ye bakiyoruz, 3 katlı çapraz doğrulama, işlemcileri full performans ile kullan, bana raporlama yap
gs = GridSearchCV(SVD,param_grid,measures=['rmse', 'mae'],cv=3,n_jobs=-1,joblib_verbose=True)

#fit ettik
gs.fit(data)

#en iyi ölçüm metrikleri
gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################


#best params içinden gelen en iyi parametreleri girmek için bu kodu yazıyoruz
svd_model = SVD(**gs.best_params['rmse'])

#tekrardan modelleme yapmadan önce bütün veriyi oluşturyoruz
data = data.build_full_trainset()
#modele sokuyoruz
svd_model.fit(data)

#tahmin normalde verdiği puan 4, bizim tahminimiz 4.20
svd_model.predict(uid=1.0, iid=541, verbose=True)

