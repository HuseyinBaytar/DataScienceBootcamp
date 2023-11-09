#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################
# ürün içeriklerinin benzerlikleri üzerinden tavsiyeler geliştirirlir.
# örnek, bir kullanıcı okuduğu kitap var, kitabın kategorisi/içeriklerine göre benzer bir kitap öneririz.

#metinleri matematiksel olarak temsil edecek şekle getirmemiz lazım. metinler vektörler aracılığıyla temsil edebilirsek
#count vector ve TF-IDF yöntemleriyle
# filmlerin açıklamalarındaki bazı kilit kelimeleri yakalayıp, diğer film açıklamalarıyla benzer olanları yakalıyoruz.
#satırlarda filmlier, sütunlarda metinler

#1. olarak metinleri sayısala çevirdik
#2. olarak euclide uzaklıklarını hesaplayıp, en düşük uzaklığın benzer olduğunu görüyoruz
#euclidean distance = bu uzaklık  // başka bir benzerlik ölçüsü olan cosine similarity = bu benzerlik



#####################################count vectorizer
#adım1: eşsiz tüm terimleri(kelime) sütunlara, bütün dokümanları(her ne ile ilgileniyosak, tweet,ürün title) satıra yerleştir.
#adım2: terimlerin dökümanlarda geçme frekanslarını hücrelre yerleştir

################################ TF - IDF (kelimelerin hem kendi metinlerinde hemde tüm odaklandığımız verideki frekansların üzerinden normalizasyon işlemi yapar)
#1. adım = count vectorizer'ı hesapla
#2. adım = TF - Term Frequency'i hesapla ( t teriminin ilgili dokumandaki frekansı / dokumandaki toplam terim sayısı)
# bahsedilen kelimenin, verilen cümledeki toplam kelime sayısına bölümü

#3. adım = IDF hesapla (inverse document frequency)
# 1+ loge((toplam döküman sayısı +1) / (içinde t terimi olan döküman sayısı +1))
# 1+ loge((toplam farklı cümlelerin sayısı + 1) / (içinde ilgili kelimenin geçtiği toplam cümle sayısı +1))

#4. adım = TF * IDF 'i hesapla,  tf ile idf'i çarparız

#5. adım =  L2 normalizasyonu yap
#(satırlarının kareleri toplamının karekökünü bul, ilgili satırdaki tüm hücreleri bulduğun değere böl


#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################
# 1. TF-IDF Matrisinin Oluşturulması
#################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("week5/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape
#veri setinden sadece overiew'e odaklanıcaz
df["overview"].head()

#tfidf yöntemini kullanıyoruz / modelini kuruyoruz
tfidf = TfidfVectorizer(stop_words="english")
#sık kullanılan kelimeleri çıkardık örneğin: and , the, on, in  bir değer taşımıyorlar

#NaN'lari boşluklarla değiştirdik / NaN'ler hesaplamalarda problem çıkarabilir
df['overview'] = df['overview'].fillna('')

#fit edip dönüştürür
tfidf_matrix = tfidf.fit_transform(df['overview'])

#45,466 film yorumları var ve 75.827 kelime var
tfidf_matrix.shape


tfidf.get_feature_names_out()

tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################
#metin vektörlerini matematiksel olarak , hangi filmlerin birbirine benzer olduğunu bulduğumuz bölüm

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape #dökümanların birbirine olan benzerlikleri
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################
#hesapladığımız skorları, değerlendirmek için isimleri alıyoruz
#her bir filmin ismi, ve indexleri olan bir serie yaptk
indices = pd.Series(df.index, index=df['title'])

#filmlerden fazla sayıda var
indices.index.value_counts()

#çoklama filmlerden 1 tanesini tutup, diğerlerini siliyoruz, güncellik olarak en sonundakini alıyoruz
indices = indices[~indices.index.duplicated(keep='last')]

#sherlock holmes filmlinin indexini tutuyoruz
movie_index = indices["Sherlock Holmes"]

#coisine_sim'e sherlockun indexiyle gidiyoruz
cosine_sim[movie_index]

#sim score diye bir df oluşturup, benzerlik olan ları al ve score olarak değerlendir
similarity_scores = pd.DataFrame(cosine_sim[movie_index],columns=["score"])

#en yüksek 10 score sahip filmleri çağırıyoruz  #0. gözlemde filmin kendisi var o yüzden 1-11 veriyoruz
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

#index bilgisi olan filmlerin title'larını alıyoruz
df['title'].iloc[movie_indices]



#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)

#cosine similatiry matrisi çıkaran fonksiyon
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

