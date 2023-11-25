################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################
#2001 yılında Leo Breimanın oluşturduğu yöntemlerden biridir.
#temeli birden çok karar ağacın ürettiği tahminlerin bir araya getirilerek değerlendirilmesine dayanır. Aşırı öğrenmeye meyilli olan yöntem CART'ı,  Random Forest ile bu problemi çözmeye çalışmıştır.Her bir ağaca fikrinin sorulup, hepsinin fikirlerinin tahminlerinin çoğunluğunu aldığı bir düzendir.
#Bagging ile random subspace yöntemlerinin birleşimi ile oluşmuştur.
#Ağaçlar için gözlemler bootstrap rasgele örnek seçim yöntemi ile değişkenler random subspace yöntemi ile seçilir.

#Karar ağaçının her bir düğümünde en iyi dallara ayırıcı(bilgi kazancı) değişken tüm değişkenler arasından rastgele seçilen daha az sayıdaki değişken arasında seçilir.

#Random forest dediğimizde rastgeleliğini nerden alıyor dediğimizde aklımıza 2 şey gelicek.
#1- gözlem birimlerinden rastgele seçerek N tane ağaç oluşturur.
#2- oluşturduğu ağaçlarda bölme işlemlerine başlamadan önce değişkenlerin içinde rastgele değişkenler seçer daha az sayıda.

#Ağaç oluşturmada veri setinin 2/3'ü kullanılır. Dışarıda kalan veri ağaçların performans değerlendirmesi ve değişken öneminin belirlenmesi için kullanılır.

#HEr düğüm noktasında rastgele değişken seçimi yapılır

#Leo Breiman 1996da çıkarttığı yöntemde, aslında CART methodunda yaptığını bir çok ağaç üzerinden yapmıştır. Gözlem seçimlerinde rastgele bir seçim işlemi yaptım ve burda bir rassalık yakaladım diye düşünmüş olmalı ki daha sonra Random Subspace yöntemini kullanarak, Değişkenlerdede rastgele seçimler yapsak buda başarılı olur diye düşünerek, gözlemlerde ve değişkenlerde rastgelelik diyerek Random Foresti oluşturmuştur.

#Diyelim ki elimizde m adet gözlem birimi var, bu gözlem birimi içinden n'er adet gözlem (n <m ) ve mden daha küçük olarak boostrap yöntemi ile seçilir yani, 1000 tane gözlem var, 750'den az olanlar sçeilir. Ağaçlar birbirinden bağımsız

#Bagging yönteminde ağaçların birbirine bağımlılıkları yoktur Boosting yönteminde ise ağaçlar artıklar üzerine kurulur dolayısıyla ağaçların birbirine bağımlılıkları vardır.
#bagging yöntemi overfitte karşı dayanıklı bir modeldir.


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("week7/datasets/diabetes.csv")
#bağımsız değişken
y = df["Outcome"]
#bağımlı değişkenler
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################
#modelimizi kuruyoruz
rf_model = RandomForestClassifier(random_state=17)

#hiper parametre araması yapmadan önce bi model accuracylerimize bakıyoruz
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


#grid search ile aramaya sokup, en iyi değerin ne olabiliceğine bakıyoruz
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#olası 180 model var, toplam 900 fit etme işlemi gerçekleşti
rf_best_grid.best_params_

#son halini bir final dosyasına atıyoruz
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
#tekrardan en iyi parametrelerde deniyoruz
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() #75 den 76ya
cv_results['test_f1'].mean()       #61 den 64'e
cv_results['test_roc_auc'].mean()  #82 den 82'ye


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
#en önemli olan feature'a bakıyoruz
plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
#maximum derinliğe göre bir değerlendirme yapıp bakabiliyoruz
val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


################################################
# GBM
################################################
#GBM artık optimizasyonuna dayalı çalışan bir ağaç yöntemidir. Ağaç yöntemlerine Boosting yöntemi ve gradient descent'in uygulanmasıdır.
#AdaBoost(Adaptive boosting) yöntemi, GBM'in temellerinin dayalı olduğu bir yöntemdir. Adaboost zayıf sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturması fikrine dayanır.

#GBM'de hatalar/artıklar üzerine tek bir tahminsel model formunda olan modeller serisi kurulur.
#ağaçlar tahminde bulunduktan sonra hatalarını düzeltip onları boostlamak için yapılır. Seri içerinsdeki bir model serideki bir önceki modelin tahmin artıklarının üzerine kurularak oluşturur.
#gbm diferansiyellenebilen herhangi bir kayıp fonksiyonunu optimize edebilen gradient descent algoritmasını kullanmaktadır.
#tek bir tahminsel model formunda olan modeller serisi additive şeklinde kurulur.

#additive modeling nedir?

#Ağaç yöntemleri bir kutuyu çizgiler çekerek bölüyor diye düşünebiliriz, Additive modelde bu çekilen çizgiyi daha hassaslaştırarak biçimlendirerek, daha başarılı tahmin yapmasını sağlıyor.
#eski tahmin sonucumuza artıklardan kalan hataları ekliyerek ve çıkararak, gerçek değere yaklaşmaya çalışyoruz.


#modelimizi kuruyoruz
gbm_model = GradientBoostingClassifier(random_state=17)

#modelimizin hiperparemetre ayarını yapmadan önceki hatasına bakiyoruz
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7591715474068416
cv_results['test_f1'].mean()
# 0.634
cv_results['test_roc_auc'].mean()
# 0.82548

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
#yukarıdaki parametreleri deniyerek, en iyi parametreyi bulmaya çalışyıoruz
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
#en iyi parametrelerimizi bulduk
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  #75den 78'e çıktı
cv_results['test_f1'].mean()        #63den 66'ya çıktı
cv_results['test_roc_auc'].mean()   #82 sabit kaldı


################################################
# XGBoost
################################################
#XGBoost, GBM'in hız ve tahmin performansını artırmak üzere optimize edilmiş, ölçeklenebilir ve farklı platformlara entegre edilebilir versiyonudur. Tiandqi Chen tarafından 2014 yılında ortaya çıkmıştır. Bir çok Kaggle yarışmasında başarısını kanıtlamış güzel bir modeldir.

#modelimizi kuruyoruz
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
#CV ile hiperparemetre optimizasyonu yapmadan önce hatalarımızı görelim
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.75265
cv_results['test_f1'].mean()
# 0.631
cv_results['test_roc_auc'].mean()
# 0.7987

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}
#colsample_bytree = değişkenlerden alınacak olan subsample sayısı
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#en iyileri hiperparemetreleri yakalıyoruz
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  #75den 76ya
cv_results['test_f1'].mean()        #63de sabit
cv_results['test_roc_auc'].mean()   #79 dan 81'e çıkmıştır


################################################
# LightGBM
################################################
#XGboostten sonra literatüre giren, bir çok yarışmada başarısını kanıtlamış bir modeldir. LightGBM, XGboos'un eğitim süresi performansını artırtmaya yönelik geliştirilen bir diğer GBM türüdür.
#level-wise büyüme stratejisi yerine leafwise büyüme stratejisi ile daha hızlıdır. XGboost geniş kapsamlı bir arama yaparken, Lightgbm derinlemesine bir arama yapmaktadır.
#Microsoft tarafından 2017 yılında yapılmıştır.

#model oluşturduk
lgbm_model = LGBMClassifier(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #74
cv_results['test_f1'].mean()        #62
cv_results['test_roc_auc'].mean()   #79


# Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #76
cv_results['test_f1'].mean()        #61
cv_results['test_roc_auc'].mean()   #82


# Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)
lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #76
cv_results['test_f1'].mean()        #61
cv_results['test_roc_auc'].mean()   #82


################################################
# CatBoost
################################################
#Category boosting'in kısaltılmışıdır. Yandex tarafından 2017 de yapılmıştır. Kategorik değişkenler ile otomatik olarak mücadele edebilen, hızlı, başarılı bir diğer GBM türevidir.

#modelimizi kuruyoruz ve hiperparemetre ayarı yapmadan accuracy bakıyoruz
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  #83
cv_results['test_f1'].mean()        #77
cv_results['test_roc_auc'].mean()   #65


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
#hiperparametre setimizi girip en iyi parametrelere bakıyoruz
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #77
cv_results['test_f1'].mean()        #63
cv_results['test_roc_auc'].mean()   #84


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
#hepsi Glucose'a önem verirken, lgbm BMI'a en çok önem vermiştir.
#toy bir veri seti ile çalışıyoruz, daha gelişmiş veri setlerinde çalışıyorken, 100lerce 1000 lerce veri setleri ile çalışıyoken, modele olan katkılarından dolayı onlarla çalışmayı tercih ederiz.
#Karar verirken, diğer modellerin önemli bulduğu glucose'a daha çok önem verilebilir

################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################
#modelimizi kuruyoruz
rf_model = RandomForestClassifier(random_state=17)
#verilecek bir hiperparametre seti içinden rastgele seçimler yapar ve en iyilerini bulmaya çalışır
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)
#fit ediyoruz
rf_random.fit(X, y)
#en iyi parametreler
rf_random.best_params_
#modelimizi fitliyoruz
rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)
#CV ile bakıyoruz
cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()      #76
cv_results['test_f1'].mean()            #62
cv_results['test_roc_auc'].mean()       #83


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################
#Auc'yi görselleştirip, overfite düşüp düşmediğimize bakıcaz
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

#belirlediğimzi parametreler
rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

#modeli kuruyoruz
rf_model = RandomForestClassifier(random_state=17)
#görselleştiriyoruz
for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]











