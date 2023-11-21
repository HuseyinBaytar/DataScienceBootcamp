################################################
# KNN
################################################
#Gözlemlerin birbirine olan benzerlikleri üzerinden tahmin yapılır. (bana arkadaşını söyle, sana kim olduğunu söylüyeyim)
#bir gözlem biriminin en kendine yakın olan diğer k adet gözlem birimi hesaplanır. Kendine en yakın olan k adet gözlem birimlerinin bağımlı değişkeni neyse ona göre tahminde bulunur.
#öklid ya da benzeri bir uzaklık hesabı ile her bir gözleme uzaklık hesabı yapılır. verilen K tane gözlemin bağımlı değişken ortalaması alınıp bilinmeyen gözlemimize veririz.
#Eğer sınıflandırma işlemi için KNN yapıcaksak, en yakın 5 gözlem biriminin ortalaması değilde, en sık tekrar eden sınıfını seçip, o değişkene atıcaz

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("week7/datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#standartscaler yapıyoruz, numpy arrayi döndürüyoruz
X_scaled = StandardScaler().fit_transform(X)
#df'e çeviriyoruz
X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################
#modelimizi kuruyoruz ve X, y'i fitliyoruz
knn_model = KNeighborsClassifier().fit(X, y)
#rastgele örneklem çekiyoruz
random_user = X.sample(1, random_state=45)
#tahmin ettiriyoruz
knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC 0.90
roc_auc_score(y, y_prob)

#5 katlı cross validate yapıyoruz
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# 0.73  accuracymiz düştü
# 0.59  f1 skore'umuz çok düştü
# 0.78  AUC score'umuz çok düştü

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################
#modelimizi kuruyoruz
knn_model = KNeighborsClassifier()
#parametrelerine bakıyoruz
knn_model.get_params()
#komşuluk sayısını değiştirerek en optimum komşu sayısını bulmaya çalışyoruz
knn_params = {"n_neighbors": range(2, 50)}

#grid search ile arıyoruz
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
#48 tane denenicek aday var, 5 katlı yapılacağı için toplam 240 tane fit işlemi vardır diyor.
#en iyi komşunun 17 olduğunu söylüyor
knn_gs_best.best_params_

################################################
# 6. Final Model
################################################
#çıkardığımız en iyi değerlerle, modelimizi güncelliyoruz
#key value şeklinde dictionary olduğunda, 2 yıldız ile atıyoruz
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

#cross validate ile hatalara bakmak için 5 katlı yapıyoruz
cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#73 den 76'ya çıktı
cv_results['test_f1'].mean()
#59 dan 61'e arttı
cv_results['test_roc_auc'].mean()
#78den 81'e arttı

#rastgele user alıp ,tahmin ediyoruz
random_user = X.sample(1)
knn_final.predict(random_user)
