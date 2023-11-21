# Logistic REgression
#amaç sınıflandırma problemi için bağımlı ve bağımsız değişkenler arasındaki ilişkiyi doğrusal olarak modellemektir.
#normal regressionun aynısı fakat çıkan sonucu sigmoid fonksiyonunda yazıcaz.tahmin edilen nihai sonuç 0 ile 1 içinde yer alsın diye sigmoid fonksiyonuna yazarız.
#nasıl? Gerçek değerler ile tahmin edilen değerler arasındaki farklara ilişkin log loss değerinin minimum yapabiliecek ağırlıkları bularak yapılır.

#logistic regression için Gradient Descent
#entropi ne kadar yüksekse çeşitlilik o kadar fazladır. entropinin düşük olmasını isteriz

#sınıflandırma problemlerinde başarı değerlendirme
#confusion matrix: True pozitif / False negatif ,,  False possitive // True NEgative
#Accuracy: Doğru sınıflandırma oranı (tp+tn) / (tp+tn+fp+fn)

#false positive = yanlış alarm gibi düşünebiliriz  1. tip hata denir
#precision: Pozitif sınıf tahminlerinin başarı oranı TP / ( TP + FP )
#false negatif: 2. tip hatadır
#Recall : Pozitif sınıfın doğru tahmin edilme oranıdır  TP / (TP + FN)
#F1 Score : 2 * (precision * Recall) / ( Precision + Recall)
#precision tahminlerin başarısına, recall değeri gerçekleri yakalama başarısına odaklanmıştır. ikiside önemlidir, f1 skoru ikisininde değerlerini tutmaktadır.

#Classification Threshold
#diyelim ki thresh holdumuz 0.50,  thresholddan yüksek olanlar 1'e, düşük olanlar 0'a yuvarlanır
#eşik değerinin değişmesine göre, accuracy, recall, f1 , precision vs hepsi değişir. bu problemi çözmek için roc curve gerekir

# ROC Curve
# olası bütün threshholdları çıkarıp, confusion matrix çıkarılır. true positive rate ile false positive rate tablosunda denk gelme kısmına göre belirlenir.
#Area Under Curve (AUC)
#roc eğrisinin tek bir sayısal değer ile ifade edilişidir. roc eğrisinin altında kalan alandır. AUC, tüm olası sınıflandırma eşikleri için toplu bir performans ölçüsüdür.

#sınıflandırma problemleri söz konusu olduğunda dikkat etmemiz gereken 1. konu, sınıf dağılımı dengesiz mi. eğer dengesizse recall, precision ve f1 score'una bakılır. sonrasında AUC değerlerine bakılır

#LOG Loss
#bir başarı metriğidir. Modelin başarısı için değerlendiririz ayriyetten, optimizasyon içinde kullanılır.
#entropi yine, ne kadar düşük o kadar iyi ne kadar yüksek o kadar kötü, çeşitlilik lazım
#örneğin bir sınıfda 10101 var, diğer sınıfta 11111,  1 ve 0'ların olduğu sınıf daha çeşitlidir, entropisi yüksektir.



######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from scikitplot.metrics import plot_roc


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("week7/datasets/diabetes.csv")

df.head()

##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()

#görselleştirerek dağılımına bakıyoruz
sns.countplot(x="Outcome", data=df)
plt.show()

#oranlarına bakıyoruz
100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df.head()

#kendimizi tekrar etmemek için fonksiyon tanımlıyarak hepsine bakiyoruz
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

#bağımlı değişkeni dışarda bırakıp yapıyoruz
cols = [col for col in df.columns if "Outcome" not in col]

for col in df.columns:
    plot_numerical_col(df, col)


##########################
# Target vs Features
##########################
#outcome'a göre groupby'a alıp, hamilelerin ortalamasına bakıyoruz
df.groupby("Outcome").agg({"Pregnancies": "mean"})
#hepsine tek tek bakmak için fonksiyon yapıyoruz
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)



######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()
df.isnull().sum()

#eksik değerler var ama göz önüne almıyoruz çünkü feature engineering ile ilgilenmiyoruz şu an
df.describe().T

#veride herhangi bir yerinde outlier var mı ?
for col in cols:
    print(col, check_outlier(df, col))
#insülin değişkenindeki aykırı değerleri bastır
replace_with_thresholds(df, "Insulin")

#tüm featureları, robust scale ile fit edip transform et
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################
# bağımlı değişkeni ayırıyoruz
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#modelimizi kuruyoruz
log_model = LogisticRegression().fit(X, y)
#bias-sabit-beta
log_model.intercept_
#ağırlıklar- weight
log_model.coef_
#model denklemini yazın diye bir soru gelirse mülakatta, intercept + x1*0.59906785 + x2 * 1.41770936
#tahmin işlemini yapıyoruz
y_pred = log_model.predict(X)
#tahminde bulunduğu ilk 10 değer
y_pred[0:10]
#gerçek ilk 10 değer
y[0:10]


######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
#confusion matrix  görüntüsü
plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65


# ROC AUC // probabilityleri
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939


######################################################
# Model Validation: Holdout
######################################################
#veri setimizi 80'e 20 bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
#modelimizi kuruyoruz
log_model = LogisticRegression().fit(X_train, y_train)
#eğitim yaparken göstermediğimiz test setini soruyoruz
y_pred = log_model.predict(X_test)
#olasılık değerlerini çıkarıyoruz
y_prob = log_model.predict_proba(X_test)

#tahmin ettiğimiz değerlerle, modelin görmediği test skorlarını karşılaştırıyoruz
print(classification_report(y_test, y_pred))

#öncekiyle çok büyük farklılıklar yok fakat, model görmediği veriye dokunduğunda daha başarısız gibi gözüküyor. model doğrulama işlemi gerekiyor.
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

y_prob.reshape(-1,1)

plot_roc(y_test, y_prob)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# AUC
roc_auc_score(y_test, y_prob)
#diğer değerimiz 83 dü şu an ki auc değerimiz 87.5 çıktı

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#modelimizi kuruyoruz, bütün veriyi vererek yapıyoruz validation'u
log_model = LogisticRegression().fit(X, y)
#bağımsız değişkenleri ver, 5 katlı cross validation diyoruz, birden fazla başarı metriğini hesaplar
cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

#hesaplanmış olan 5 tane test score'unun ortalamasına gidiyoruz
cv_results['test_accuracy'].mean()
# Accuracy: 0.7721
#hesaplanmış olan 5 tane precision'un ortalamasına bakıyoruz
cv_results['test_precision'].mean()
# Precision: 0.7192
#hesaplanmış olan 5 tane recall'in ortalamasına bakıyoruz
cv_results['test_recall'].mean()
# Recall: 0.5747
#hesaplanmış olan 5 tane test_f1'in ortalamasına bakıyoruz
cv_results['test_f1'].mean()
# F1-score: 0.6371
#hesaplanmış olan 5 tane roc auc' ortalamasına bakıyoruz
cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns
#random bir kullanıcı seçtik
random_user = X.sample(1, random_state=45)
#tahmin ediyoruz. bu kişi diyabettir diyor.
log_model.predict(random_user)


