########################################################Collaborative Filtering ( iş birlikçi filtreleme )#######################################################
# - Item Based Collaborative Filtering                     #memory base
# - User Based collaborative Filtering                     #memory base
# - Model Based collaborative Filtering                    #latent factor model


###########################################
# Item-Based Collaborative Filtering
###########################################
#item benzerliği üzerinden öneriler yapılır.
# örn: izleyici bir filmi beğendi,  'izlenen filmin beğenilen beğenilme yapısına benzeyen başka bir film önerilir'

#işbirlikçilik = bir toplumun verdiği oylar, verilen puanlar. başka bir bir filmin verilen puanlarına benzer, corelasyon var ise diğer film önerilir


######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


movie = pd.read_csv('week5/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('week5/datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################
#örneğin bir kullanıcı, 1 filme puan verdi, ama diğer tonlarca filme puan vermediği için çok fazla hücre temsil edecektir, performans problemi çıkarır
#çeşitli indirgemeler yapılmalıdır. örneğin 1000'den az sayıda rate almış filmleri çalışmadan dışarıda bırakabiliriz.

df["title"].nunique()
#27.262 unique film  var

#hangi filme kaçar tane yorum gelmiş?
df["title"].value_counts().head()

#her bir title'dan kaç defa geçmişi, dataframe'e çeviriyoruz
comment_counts = pd.DataFrame(df["title"].value_counts())

#1000'den küçük yorumu olan filmleri rare moviese atıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index


#az rate alan filmlerden kurtulmak için, tilda ile 1000'den büyük comment countu olanları aldık
common_movies = df[~df["title"].isin(rare_movies)]
#17 milyon rate var
common_movies.shape

#3159 eşsiz film var
common_movies["title"].nunique()

#veri setimizin ilk halinde 27 bin tane film vardı
df["title"].nunique()

#satırlarda kullanıcılar, sütunlarda title'lar olarak bir pivot table ile  indirgemiş olduğumuz yeni veri setini yazdırıyoruz. kesişim olarakta ratingleri verdik
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################
#elimizdeki matris, sütunlarında film isimlerini, satırlarında oylayan kullanıcıları var

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"

#verilen film ismine göre corelasyonu yüksek olan ilk 10'u verdik
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


#rastgele film seçip deneme yapmak için
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

#fonksiyon ile, verilen keyworldü girdiğimizde, title'larda gezip eğer keywordu bulursa bize getirir
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





