############################################
# User-Based Collaborative Filtering
#############################################
#kullanıcı(user) benzerlikleri üzerinden öneriler yapılır.


#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('week5/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('week5/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#pd serisi oluşturup, rastgele 1 tane örneklem çekiyoruz
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

#user movie dfden bu kullanıcıyı seçiyoruz
random_user_df = user_movie_df[user_movie_df.index == random_user]

#NaN'lar puan verilmediğini ifade eder, o yüzden nan olmayanları alıyoruz
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#indexe kişiyi yazıyoruz, sütunlarınada filmi yazıp kontrol ediyoruz
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]
#izlemiş ve 1 puan vermiş

#toplam 33 film izlemiş
len(movies_watched)

#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
#user df'e izlenen filmler listesini göndererek  sadece izlenen filmlere ilişkin bilgi ederiz
movies_watched_df = user_movie_df[movies_watched]
#33 film ve 138bin kişi

#soru şu: user ile 1 tane bile aynı filmi izliyen kişi ile, 33 izlediği filmin belki 10-15 tanesini izliyen kullancıların aynı listeye girmesi gerekir mi ?
#her bir kullanıcının kaç tane film izlediğine gidicez
user_movie_count = movies_watched_df.T.notnull().sum()

#indexde yer alan userid'yi  değişkene çeviriyoruz
user_movie_count = user_movie_count.reset_index()

#değişkenleri isimlendirdik
user_movie_count.columns = ["userId", "movie_count"]

#130bin kullanıcıdan , 3 bin kullanıcı kaldı.
#userımız ile en az 20 tane film izliyen kişilere erişiyoruz
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

#17 kişi 33 de 33 yapmış
user_movie_count[user_movie_count["movie_count"] == 33].count()

#%60'ı en azından izlemiş olsun diye 20'den büyüğü referans aldık(şahsi kişisel yorum) ve kullanıcıların id'lerini aldık
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]


#biraz daha programatik şekilde yapmak için, userın izlediği filmlerin boyutunu alıp, %60'ını alıp, oranı yazıyoruz.

# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100


#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. user ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# veri setlerini bir araya getirdik
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

#final df'in transpozunu alıp, corelasyonunu hesaplayip pivotunu alıyoruz ve duplike kayıtları çıkarıyoruz
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

#dataframe'e çeviriyoruz ve ismini değiştiriyoruz ve indexlerini resetliyoruz / daha okunabilir bi forma getiriyoruz
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()


#user id 1 önceden belirlediğimizi , korelasyonuda %65 den büyük olanları getiriyoruz
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

#bizim önceden belirlediğimizi atıp, korelasyonu en yüksekleri sıraladık, ve sütünları tekrar isimlendirdik
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

#rating dosyasını okutup, elimizdeki top users df'ini birleştirip, bize userid,movieid ve ratingi istediğimizi söyledik
rating = pd.read_csv('week5/datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

#ilk seçtiğimiz kişi kendine 1 korelasyon çıkacaktır, oy üzden onu listemizden çıakrıp tekrar yazdırdık
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#elimizdeki tabloda user'ımızda en yüksek korelasyona sahip olan kullanıcılar ve bunların çeşitli filmlere verdiği puanlar var,
#userımız ile en az 20 tane aynı filmi izliyen insanların verdiği oylar

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
#problemimiz bir skor problemi, referans noktası problemi. kullanıcının verdiği puanımı göz önünde yada correlasyonu mu ?
#yapıcağımız işlemde hem korelasyonu hemde ratinge değer verip sıralama yapıyoruz
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

#hem rating hemde corelasyon etkisi olan bir değere göre sıralıyoruz
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

#df'e kaydediyoruz
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#index resetliyoruz
recommendation_df = recommendation_df.reset_index()

#ağırlıklı ratingi 3.5'den büyük olanları sıralıyoruz//  3.5 yorumdur isteğe göre değişebilir
recommendation_df[recommendation_df["weighted_rating"] > 3.5]

#öneri olarak, 3.5'dan büyük olanları  büyükten küçüğe olarak sıralayıp önerilecek olan listemizi çıkarıyoruz
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


#film isimlerini getiriyoruz.
movie = pd.read_csv('week5/datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])


#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

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

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#otomatik fonksiyon, yüzde, ve threshold corelasion ile veriyoruz, ve verilen score'a göre sıralıyoruz
def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])


#random birini seçtik
random_user = int(pd.Series(user_movie_df.index).sample(1).values)
#fonksiyondan döndürüp bu kullanıcıyla benzer şeyleri getirdi
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


