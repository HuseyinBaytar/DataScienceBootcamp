##################################################
# Smoothing Methods (Holt-Winters)
##################################################
#Bildiğimiz bütün yöntemleri optimizasyon yöntemi ile deneyip, en düşük hatayı veren en iyi modeldir diyeceğiz.



import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
warnings.filterwarnings('ignore')


############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

#loading data from statsmodel
data = sm.datasets.co2.load_pandas()
#making our time serie data
y = data.data
#aylık olarak tahmin yapmak daha mantıklı olacaktır, haftalık formattan aylık formata çeviriyoruz.
y = y['co2'].resample('MS').mean()
#5 tane boş değerimiz var
y.isnull().sum()

#kendisinden önceki yada sonraki değerlerle doldurulabilir, zaman serilerindeki eksik değerler. 6. aydaki değerimiz nan'dı, 7. aydaki değeri 6. ayla doldurmuş olduk.
y = y.fillna(y.bfill())


y.plot(figsize=(15, 6))
plt.show()
#bu seride trend vardır. Durağan değildir. mevsimsellikte vardır.


############################
# Holdout
############################
#veri setini 2ye ayırıyoruz
train = y[:'1997-12-01']
len(train)  # 478 ay
# 1998'ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay
#overfitingi engellemek için ve elde ettiğimiz hataları daha doğru değerlendirmek için train ve teste ayırdık.

##################################################
# Zaman Serisi Yapısal Analizi
##################################################
# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)
#hipotez testimizi yapıyoruz ve p value'muz 0.999 çıkıyor, yani seri durağan değildir.

# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)
#toplamsal modelin bileşenleri ortaya çıktı, ilk sıramız level, serinin değerlerini gösterdi, trend yayrılmış, mevsimselliği var, son kısımdada artıkları görüyoruz, 0'ın etrafında olmasını bekleriz meanininin
#serinin durağan olup olmadığını söyleyen bir çıktıda alabiliyoruz.

##################################################
# Single Exponential Smoothing
##################################################
#Sadece durağan serilerde başarılıdır. Trend ve mevsimsellik olmamalı.
#üstsel düzeltme yaparak tahminde bulunur.
#gelecek yakın geçmişle daha fazla ilişkidir varsayımıyla geçmişin etkileri ağırlıklandırılır.
#geçmiş gerçek değerler ve geçmiş tahmin edilen değerlerin üssel olarak ağırlıklandırılmasıyla tahmin yapılır.

# SES = Level
#modelimizi kuruyoruz/ alpha değerini 0.5 girdik, eğer girmeseydik simpleexp smoothing kendiliğinden doğru şekilde buluyor olucaktı.
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
#tahminlerimizi yapıyoruz/ predict değil forecasti kullanıyoruz
y_pred = ses_model.forecast(48)
#5.70 error alıyoruz
mean_absolute_error(test, y_pred)


#görselleştiriyoruz, yeşiller tahmin ettiğimiz değerler. gördüğümüz üzere çok kötü tahmin
train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()


#bu görselli sık sık kullanıcağımız için bir fonksiyon tanımlıyorum
def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

#parametrelere bakabiliyoruz
ses_model.params

############################
# Hyperparameter Optimization
############################
#farklı alpha değerleriyle hatamızın kaç olduğunu görüp, en iyi değeri bulmaya çalışıyoruz
def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)
#0.8 den 0.001 şekilde artacak şekilde 1 'ek adar bir sürü değer oluşturduk
#ses formulasyonu
# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka
ses_optimizer(train, alphas)

#best alphayı ve best mae'yi yakalıyoruz
best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################
#final modelimizi kuruyoruz
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")
#yeşil yani tahmin ettiklerimiz düzelmedi ama azda olsa yaklaştı

##################################################
# Double Exponential Smoothing (DES)
##################################################
#trend etkisini göz önünde bulundurarak üssel düzeltme yapar
# DES: Level (SES) + Trend
#temel yaklaşım aynıdır. SES'e ek olarak trendde dikkate alınır.
#Trend içeren ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.

#üstte gördüğümüz toplamsal model, altta gördüğümüz çarpımsal modeldir.  çarpımsal bir seri, fonskyionda çarpımsal ifadeler olduğunu ifadesini taşır. Mevsimsel ve artık bileşenler trendden bağımsızsa
#bu durumda seri toplamsal seridir, eğer bağığmsız değilde çarpımsal seridir.
#mevsimsel ve artıklar 0'ın etrafında dağılıyorsa toplamsaldır diyebiliriz.

# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise

#modelimizi kuruyoruz ve trendi ekliyoruz
des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_trend=0.5)
#tahminlerimizi yapıyoruz
y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")
#yeşil çizgimiz bizim tahmin ettiklerimiz, yukarı doğru çıkmış gözüküyor


############################
# Hyperparameter Optimization
############################
#en iyi parametreleri arıyoruz
def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

############################
# Final DES Model
############################
#en iyi parametrelerle final modelimizi kuruyoruz
final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta)
#tahminlerimizi yapıyoruz
y_pred = final_des_model.forecast(48)
#görselleştirdiğimiz zaman yeşil çizginin tam ortada olduğunu görüyoruz, mevsimselliği yakalıyamadık.
plot_co2(train, test, y_pred, "Double Exponential Smoothing")

##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################
#Triple exponential smoothing en gelişmiş smoothing yöntemidir.
#Bu yöntem dinamik olarak level, trend ve mevsimsellik etkilerini değerlendirerek tahmin yapmaktadır.
#Trend veya mevsimsellik içerek tek değişkenli serilerde kullanılabilir.
# TES = SES + DES + Mevsimsellik

#modelimizi kuruyoruz
tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)
#tahmin ettiriyoruz
y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")
#biraz daha tutarlı bir sonuç aldık, ama hala iyi değil bir parametre optimizasyonu yapmamız lazım

############################
# Hyperparameter Optimization
############################

#3 tane kombinasyon üretiyoruz
alphas = betas = gammas = np.arange(0.20, 1, 0.10)
#listeye çeviriyoruz
abg = list(itertools.product(alphas, betas, gammas))

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################
#elimizdeki en iyi değerler ile final modeli kuruyoruz
final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)
#tahmin ettiriyoruz
y_pred = final_tes_model.forecast(48)
#görselleştirdiğimiz zaman çok güzel bir sonuç aldığımızı görüyoruz
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

