######################  iş problemi  ######################

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

###################### Veri Seti Hikayesi #######################
#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin
# hizmetlerinden ayrıldığını, kaldığını ve ya hizmete kaydolduğunu gösterir.

#CustomerId          = Müşteri İd’si
#Gender              = Cinsiyet
#SeniorCitizen       = Müşterinin yaşlı olup olmadığı(1, 0)
#Partner             = Müşterinin bir ortağı olup olmadığı(Evet, Hayır)
#Dependents          = Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır)
#tenure              = Müşterinin şirkette kaldığı ay sayısı
#PhoneService        = Müşterinin telefon hizmeti olup olmadığı(Evet, Hayır)
#MultipleLines       = Müşterinin birden fazla hattı olup olmadığı(Evet, Hayır, Telefonhizmetiyok)
#Internet Service    = Müşterinin internet servis sağlayıcısı(DSL, Fiber optik, Hayır)
#Online Security     = Müşterinin çevrimiçi güvenliğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
#Online Backup       = Müşterinin online yedeğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
#Device Protection   = Müşterinin cihaz korumasına sahip olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
#TechSupport         = Müşterinin teknik destek alıp almadığı(Evet, Hayır, İnternet hizmetiyok)
#StreamingTV         = Müşterinin TV yayını olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
#Streaming Movies    = Müşterinin film akışı olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
#Contract            = Müşterinin sözleşme süresi(Aydan aya, Bir yıl, İkiyıl)
#Paperless Billing   = Müşterinin kağıtsız faturası olup olmadığı(Evet, Hayır)
#PaymentMethod       = Müşterinin ödeme yöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik), Kredikartı(otomatik))
#MonthlyCharges      = Müşteriden aylık olarak tahsil edilen tutar
#Total Charges       = Müşteriden tahsil edilen toplam tutar
#Churn               = Müşterinin kullanıp kullanmadığı(Evet veya Hayır)

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("week7/datasets/Telco-Customer-Churn.csv")
df.head()
df.describe([0.01, 0.99]).T
##########     Görev 1 : Keşifçi Veri Analizi

#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]             # kategorik değişkenleri yakala
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]                                                #numeric ama kategorik değişkenler
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]                                                #kategorik olup bir bilgi taşımayan yani çok fazla eşsiz olan
    cat_cols = cat_cols + num_but_cat                                                          #tüm kategorik değişkenleri topluyoruz
    cat_cols = [col for col in cat_cols if col not in cat_but_car]                             #kardinal olmayanları yakala

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]           # numerikleri yakala,
    num_cols = [col for col in num_cols if col not in num_but_cat]                          # numeric ama categoric olanları alma

    print(f"Observations: {dataframe.shape[0]}")                        # shape'in iyazdır
    print(f"Variables: {dataframe.shape[1]}")                             #variableların shape'ini
    print(f'cat_cols: {len(cat_cols)}')                                  #kategorikleri yazdır
    print(f'num_cols: {len(num_cols)}')                                 #numerikleri yazdır
    print(f'cat_but_car: {len(cat_but_car)}')                           #cardinalleri yazdır
    print(f'num_but_cat: {len(num_but_cat)}')                           #numeric ama kategorik olanları yazdır
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df.info()
df["SeniorCitizen"].value_counts()
df["SeniorCitizen"]= df["SeniorCitizen"].astype("O")

df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = df['TotalCharges'].astype(float)
#11 tane değişken boşluk olarak verilmiş
df["TotalCharges"].value_counts()

df["Churn"] = df["Churn"].map({"No": 0, "Yes" : 1})


#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def category_summary(dataframe, col_name, plot=False):
    print("###########################")
    print(f"{col_name} : {dataframe[col_name].unique()}")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe);
        plt.xticks(rotation=90);
        plt.figure(figsize=(10, 10));
        plt.show(block=True);

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        category_summary(df, col, plot=False);
    else:
        category_summary(df, col, plot=False);


def number_summary(dataframe, numberical_col, plot=False):
    quantiles = [0, 0.05, 0.95, 0.99, 1]
    print(dataframe[numberical_col].describe(quantiles).T)
    if plot:
        dataframe[numberical_col].hist(bins=15,ec='white')
        plt.xlabel(numberical_col)
        plt.title(f"Frequency of {numberical_col}")
        plt.show(block=True)

for col in num_cols:
    number_summary(df, col, plot=False)

#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.


def target_summary_with_cat(dataframe, target, categorical_col):
    if target==categorical_col:
        print("no")
    else:
        print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


#Adım 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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

for col in num_cols:
    print(col, check_outlier(df, col))

#Adım 6: Eksik gözlem var mı inceleyiniz.
def missing_values_table(dataframe, na_name=False): #eğer true dersek bize eksik değişkenlerin isimlerini verir
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]   # eksik değişkenlerin olduğu colunmsu yakala
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)   #eksik değer sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)      #eksik değer oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])       # birleştir
    print(missing_df, end="\n")  #df'i yazdır
    if na_name:
        return na_columns

missing_values_table(df)


#########      Görev 2 : Feature Engineering

#Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df.isnull().sum()


#Adım 2: Yeni değişkenler oluşturunuz.


#Cinsiyet ve SeniorCitizen'ın birleşimi
df['GenderSeniorCitizen'] = df['gender'] + df['SeniorCitizen'].astype(str)
print(df.groupby("GenderSeniorCitizen").agg({"Churn":"mean"}).to_markdown())


montly_quantiles = df["MonthlyCharges"].quantile([0,0.25,0.50,0.75]).to_list()
montly_quantiles.append(np.inf)
montly_quantiles[0] = -np.inf
df["Segment"] =  pd.cut(df["MonthlyCharges"],bins=montly_quantiles, labels=["D","C","B","A"]).astype("object")

print(df.groupby(by="Segment").agg({"Churn":"mean"}).to_markdown())

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#Adım 3:  Encoding işlemlerini gerçekleştiriniz.

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


#########       Görev 3 : Modelleme

#Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

y = df["Churn"]
X = df.drop("Churn", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

svm_model = SVC(kernel='linear',probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict_proba(X_test)[:,1:]
roc_auc_svm = roc_auc_score(y_test, y_pred)
#0.82
print(classification_report(y_test,svm_model.predict(X_test)))
#0.79

###################

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict_proba(X_test)[:,1:]
roc_auc_tree = roc_auc_score(y_test, y_pred_tree)
print(f"Decision Tree Model Accuracy: {roc_auc_tree}")
#0.65
print(classification_report(y_test,tree_model.predict(X_test)))
#0.72


###########################3
df["Churn"].value_counts()
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict_proba(X_test)[:,1:] #[:,:1]
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest Model ROC: {roc_auc_rf}")
#0.82
print(classification_report(y_test,rf_model.predict(X_test)))
#0.79

#
# 1-rf_model.predict_proba(X_test)[:,:1]
##############################333

from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(max_iter=500)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict_proba(X_test)[:,1:]
accuracy_logreg = roc_auc_score(y_test, y_pred_logreg)
print(f"Logistic Regression Model Accuracy: {accuracy_logreg}")
#0.84
print(classification_report(y_test,logreg_model.predict(X_test)))
#0.81

##############################3

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=34)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict_proba(X_test)[:,1:]
roc_auc_knn = roc_auc_score(y_test, y_pred_knn)
print(f"K-Nearest Neighbors Model Accuracy: {roc_auc_knn}")
#0.83
print(classification_report(y_test,knn_model.predict(X_test)))
#0.78


#Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.

from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

knn_model.get_params()
knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

best_knn_model = knn_gs_best.best_estimator_

cv_results = cross_validate(best_knn_model,
                            X,
                            y,
                            cv=5,
                            scoring=["roc_auc"])

cv_results['test_roc_auc'].mean()
#0.83