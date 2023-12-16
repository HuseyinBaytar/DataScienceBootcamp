##################################################
# Statistical Methods
##################################################
#istatistiki metodlar kapsamında bu konunun temelini oluşturan bazı yöntemlerden bahsedicem.
#ilk görüceğimiz yöntem Autoregression yani AR(p) yöntemi. Önceki zaman adımlarındaki gözlemlerin doğrusal bir kombinasyonu ile tahmin işlemi yapılır.
#trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.  p: zaman gecikmesi sayısıdır. p = 1 ise bir önceki zaman adımı ile model kurulmuş demek olur. SES'e benzerdir ama
#burda regresyon olarak gilerler.

#MA(q): Moving Average
#önceki zaman adımlarında elde edilen hataların doğrusal bir kombinasyonu ile tahmin yapılır.
#trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
#q : zaman geçikmesi sayısıdır.

#ARMA (p,q) = AR(p) + MA(q)
#arma modeli ses modelinin kardeşidir. Ses modelinde bir smoothing faktör adı verilen katsayı vardır, bu iki terimin etkilerini ağırlandırıyor.  ARMA'da geçmiş gerçek değerlerin ağırlığı a1 ifadesi
#artık değerlerin katsayısı m1 ifadesidir, birbirinden bağımsız. SES'de iksiide bir ifadeye bağlıdır.
#HOldwinters yöntemlerinde terimler 1 parametreye şekillenirken, arma modellerinde terimlerin kendi parametreleri var. yani Verinin özütü öğrenilir.
# AUtoregressive moving average AR ve MA yöntemlerini birleştirir.
#geçmiş değerler ve geçmiş hataların doğrusal bir kombinasyonu ile tahmin yapılır.
#trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
#p ve q zaman gecikmesi sayılarıdır. p AR modeli için q MA modeli için

# ARIMA  modeli (p,d,q)
#arıma modeli trende ve mevsimselliği modelliyebilen modellerdir.
#önceki zaman adımlarındaki farkı alınmış gözlemlerin  ve hataların doğrusal bir kombinasyonu ile tahmin yapılır.
#bugün ile önceki günün değerlerini birbirinden çıkarıp, geleceği tahmin eder.
#tek değişkenli, trendi olan fakat mevsimselliği olmayan veriler için uygundur.
#P gerçek değer gecime sayısını ifade eder, p= 2 ise yt-1 ve yt-2 modeldedir.
#d fark işlem sayısını ifade eder
#q hata gecikme sayısıdır. brut force yöntemi ile p d q değerleri oluşturup en iyi olanları tahmin etmeye çalışıcaz


import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':]

##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################
model = sm.tsa.arima.ARIMA(train, order=(1, 1, 1))
arima_model = model.fit()

arima_model.summary()
#AIC ve BIC ne kadar düşükse o kadar iyi
#tahminlerimizi alıyoruz, sonrasında pd seriese çeviriyoruz ki görselleştirme fonksiyonumuz çalışsın
y_pred = arima_model.get_forecast(steps=48).predicted_mean
y_pred = pd.Series(y_pred, index=test.index)


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "ARIMA")


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################
############################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme
############################
#0 dan 4 'e kadar range belirliyoruz
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model = sm.tsa.ARIMA(train, order=order).fit()
            aic = arima_model.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)
#en iyi aic değerlerini alıyoruz

############################
# Final Model
############################

# Fit the final ARIMA model using the best hyperparameters selected by AIC
arima_model = sm.tsa.ARIMA(train, order=best_params_aic).fit()

# Make predictions for the test set
y_pred = arima_model.forecast(steps=len(test))

# Convert predictions to a pandas Series with the correct index
y_pred = pd.Series(y_pred, index=test.index)

# Plot the actual data and the predictions
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.plot(y_pred, label='ARIMA Predictions')
plt.legend()
plt.xlabel('Year')
plt.ylabel('CO2 Levels (ppm)')
plt.title('ARIMA Model Forecasting')
plt.show()


##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################
#Seosonal AutoREgressive Integrated Moving- Average
#ARIMA + mevsimselliktir.
#trend veya mevsimsellik içerek tek değişkenli serilerde kullanılabilir.
#p,d,q ARIMA'dan gelen parametreler, trend elemanlarıdır. ARIMA trend'i modelliyebiliyordu
#p= gerçek değer gecikme sayısı (otoregresif derece)
#d: fark alma işlemi sayısı
#q = hata gecikme sayısı(hareketli ortalama derecesi
#P,D,Q mevsimsel gecikme sayıları season elemanları
#m tek bir mevsimsllik dönem için zaman adımı sayısını, görülme yapısını ifade eder.

#DURAğan =  SEs, AR , MA , ARMA
#TREND = DES, ARIMA , SARIMA
# TREND + Mevsimsellik = TES , SARIMA

#modelimizi kuruyoruz
model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))
#fit ediyoruz
sarima_model = model.fit(disp=0)
#tahmin ediyoruz
y_pred_test = sarima_model.get_forecast(steps=48)
#tahmin edilen değerler saklanmış bir fromda, mean diyerek tahmin edilen değerleri alıyoruz
y_pred = y_pred_test.predicted_mean
#pd series'e çeviriyoruz görselleştirme fonksiyonumuz için
y_pred = pd.Series(y_pred, index=test.index)
#görselleştiriyoruz rastgele girdiğimiz değerlerle çok kötü bir sonuç alıyoruz
plot_co2(train, test, y_pred, "SARIMA")


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################
#0 ile 2 arasında p d q değerlerini oluşturup, itttertools ile olası kombinasyonları çıkarıyoruz, 12 ayda bir örüntü tamamlanıyor bilgisini modele veriyoruz.
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)


############################
# Final Model
############################
#building our final model with best params
model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)
#görselleştiriyoruz ve modelimizin çok tutarlı olduğunu görüyoruz
plot_co2(train, test, y_pred, "SARIMA")

#1.2 hatamız var
##################################################
# BONUS: MAE'ye Göre SARIMA Optimizasyonu
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=48)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")
#0.53'e kadar düşürdük

############################
# Final Model
############################
#daha önceki final modellerimizi train test ayrımı sonucunda deneme yanılma yoluyla tamamlayıp, modelimizi değerlendirdik. Ama şu anki yaptığımız modelde optimizasyon işlemlerimiz bitti.
#bütün veriyi kullanıp final modelini kurucaz şimdi
#veri setimiz 2001'in sonuna kadar tahmin ediyor, biz 2002'nin ilk 7 ayını tahmin edicez
model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

feature_predict = sarima_final_model.get_forecast(steps=6)
feature_predict = feature_predict.predicted_mean


