######################Linear regression (doğrusal regression)
#amaç: bağımlı ve bağımsız değişken/değişkenler arasındaki ilişkiyi doğrusal olarak modellemektir.

#veriden öğrenmemiz gereken şeyler verinin özütü, örüntüsü. bu örüntüyü temsil eden şeyler bu ifadedeki b ve W'dur.
#yi = b + wxi
#b = beta,bias,intercept
#w = katsayı, ağırlık, weight, coeficent

#diyelimki çoklu değişkenli bir regresyon analizimiz var
#yi = b  +wx1 +wx2+ wx3
#burdaki wx1 = binanın metrekaresi , wx2 binanın yaşı, wx3 binanın metroya uzaklığı.
#wx1 pozitif olması gerekir, wx2 değer büyüdükçe fiyat düşer, wx3 değer büyüdükçe fiyat düşer.
#elimizdeki veride bu ağırlıklara göre bir öğrenme sağlar
#diyelim ki yeni bir ev geldi, evin ağırlıklarını yani bina metrekaresi, yaşı, uzaklığı vs girince, önceki evlere göre bir fiyat tahmini yapar

##################### Ağırlıkların bulunması
#gerçek değerler ile tahmin edilen değerler arasındaki farkların karelerinin toplamını/ortalamasını minimum yapabiliecek b ve w değerlerini bularak.
#MSE fonksiyonuna 2m ekleyip COST fonksiyonu tanımlarız

## ######################REgresyon modellerinde başarı değerlendirme (MSE, RMSE, MAE)
#MSE: gerçek değerler ile tahmini değerlerin farkını aldıktan sonra kare alır. Kare almasının sebebi ölçüm problemini kaldırmaktır.
#ben bir tahmin yaptığımda, yapmam beklenen ortalama hatadır MSE,  en kötü senaryoda bu mse değerinin bağımlı değişkenin ortalamasına yakın olması kabuledilebilir.
#fiyat - fiyat tahmin = hata,  hata'nın karelerinin toplamının ortalaması = MSE

#RMSE: MSE'dekinin karekök alınmış halidir.

#MAE: fiyat ve fiyat tahminlerinin  farkları alınır toplanıp, n'e bölünür . MAE olur.

#bunların her biri ayrı metrikler birbiriyle kıyaslamak mantıksız. 3'üde kullanılabilir. sadece birine dayanarak model genişlettirilmeli.

############ Parametrelerin Tahmin edilmesi(ağırlıkların bulunması)
#bana en küçük hatayı verecek olan bias ve weighti bulmaktır
#cost = mse
#amacımız olası kombinasyonları deniyerek, hata fonksiyonundaki minimum noktaya gelmeye çalışmaktır.

#1 -bias ve weighti bulmanın bir yolu, analiktik çözüm olan: Normal denklemler yöntemi ( En Küçük kareler Yöntemi). ALS.

#neden istatistikte yaygınca kullanılan normal denklemler yöntemi değilde, gradient descent'i kullanalım ki?
#çünkü çoklu doğrusal regresyon yönteminde, en küçük kareler yöntemindeki final matris çözümünün tersini alma işlemi gözlem sayısı ve değişken sayısı fazla olduğunda zorlaşmaktadır.

#2- optimizasyon yöntemi: Gradient Descent
#parametrenin değerlerini iteratif bir şekilde değiştirerek çalışır.

#b0 = bias = teta0
#b1 = weight = teta1

#Doğrusal regresyon için Gradient Descent
#repeat until convergence. Türev neticesinde elde edilen gradyanın negatifine doğru giderek, ilgili parametre değerini günceller ve fonksiyonu minimum yapabilicek parametre değerine erişmeye çalışır
#Gradyanın negatifi olarak tanımlanan 'en dik iniş' yönünde iteratif olarak parametre değerlerini güncelleyerek ilgili fonksiyonun minimum değerini verebilecek parametreleri bulur.
#cost fonksiyonunu minimize edebiliecek parametreleri bulmak için kullanılır.

#learning rate'i küçük almak daha iyi olabilir. büyük alırsak minimum değeri atlıyarak gidebilir



######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
#veri setini okuyoruz
df = pd.read_csv("week7/datasets/advertising.csv")
df.shape
df.head()

#regresiyon için 2 değişken seçiyoruz
X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################
#modeli kuruyoruz
reg_model = LinearRegression().fit(X, y)

#modelin fonksiyonu
# y_hat = b + w*TV

# sabit (b - bias - intercept )
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

#televizyonun max'ı 296 ama biz 500 girdik, modelimiz satışın max'ının ilerisinde neler olabiliceğini tahmin etti.
df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

# Mean squared error
#bağımsız değişkenleri modele sorup, bağımlı değişkeni tahmin ettirdik.
y_pred = reg_model.predict(X)
#ortalama hatamızı soruyoruz
mean_squared_error(y, y_pred)
# 10.51
#y'nin ortalamasına bakıyoruz 14
y.mean()
#y'nin standart sapmasına bakıyoruz 5yani 9 ile 19 arasında değerler değişiyor gibi gözüküyor. 10 mse yüksek gibi gözüküyor.
y.std()

# RMSE /// mse'den gelen ifadenin kareköküdür.
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE // mutlak hatamıza bakıyoruz.
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE #regresyon modeline bir skor hesapla diyoruz, bu değer veri setindeki  bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir
reg_model.score(X, y)


######################################################
# Multiple Linear Regression
######################################################
#bir değil birden fazla bağımsız değişkenimiz var

df = pd.read_csv("week7/datasets/advertising.csv")

X = df.drop('sales', axis=1)

y = df[["sales"]]


##########################
# Model
##########################
#veri setimizin %20'sini test, geri kalanını train olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

#modelimizi kuruyoruz
reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

#daha fonksiyonal bir şekilde yapmak için
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)


# Test RMSE, test setinin üzerinden tahmin ettiriyoruz
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)

# 10 Katlı CV RMSE // cross val score'umuz hesaplanıyor bütün veride, score'u negatif ortalama hatayı istiyoruz diyip tilda ile çarptık, yani eksi değerler +'ya döndü/ sonra karekökünü alıp ortalamasını aldık
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 1.71

#regresyon kullanma ihtiyacı olduğunda, veri seti okuma, önişleme, modeli kurmadan önce modeli ayırmak yada tüm seti almak, modelleme işlemi ve sonrasında hata değerlendirme işlemleri.
######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################
#iteratif olarak gradient descenting çalışmasını anlamaya çalışıcaz


# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("week7/datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
