##########################  Veri Seti Hikayesi

#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde
#yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet
#test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

#########################  İŞ PROBLEMİ

#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan
# veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir


#Pregnancies                   =Hamilelik sayısı
#Glucose Oral                  =glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Blood Pressure                =Kan Basıncı (Küçük tansiyon) (mm Hg)
#SkinThickness                 =Cilt Kalınlığı
#Insulin                       =2 saatlik serum insülini (mu U/ml)
#DiabetesPedigreeFunction      =Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
#BMI                           =Vücut kitle endeksi
#Age                           =Yaş (yıl)
#Outcome                       =Hastalığa sahip (1) ya da değil (0

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("week6/diabetes.csv")
########################  Görev 1 : Keşifçi Veri Analizi
#Adım 1: Genel resmi inceleyiniz.
def checkdf(df):
    print(df.head())
    print("-----------------------------------------------------------------------------------")
    print(df.describe([0.01,0.25,0.75,0.99]).T)
    print("-----------------------------------------------------------------------------------")
    print(df.isnull().sum())

checkdf(df)

def take_a_look(df):
    if df.empty: return "Data Frame is Empty"
    print(pd.DataFrame({'Rows': [df.shape[0]], 'Columns': [df.shape[1]]}, index=["Shape"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.dtypes, columns=["Type"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.nunique(),columns=["Number of Uniques"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.isnull().sum(),columns=["NaN"]).to_markdown())
    print("\n")
    print(df.describe([0.01, 0.25, 0.75, 0.99]).T.to_markdown(numalign="right",floatfmt=".1f"))
take_a_look(df)


#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cats = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cats
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cats]
    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cats)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
#kategorik değişken yok
def category_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################")
    print(f"{col_name} : {dataframe[col_name].unique()}")
    print("###########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe);
        plt.xticks(rotation=90);
        plt.figure(figsize=(14, 14));
        plt.show(block=True);


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        category_summary(df, col, plot=True);
    else:
        category_summary(df, col, plot=True);

def number_summary(dataframe, numberical_col, plot=False):
    quantiles = [0, 0.05, 0.95, 0.99, 1]
    print(dataframe[numberical_col].describe(quantiles).T)
    if plot:
        dataframe[numberical_col].hist(bins=15,ec='white')
        plt.xlabel(numberical_col)
        plt.title(f"Frequency of {numberical_col}")
        plt.show(block=True)

for col in num_cols:
    number_summary(df, col, plot=True)





#Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
def target_summary_with_cat(dataframe, target, categorical_col):
    if target==categorical_col:
        print("no")
    else:
        print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

def target_summary_with_num(dataframe, target, numerical_col):
    if target==numerical_col:
        print("no")
    else:
        print(pd.DataFrame({f"Summary of {numerical_col}": dataframe.groupby(target)[numerical_col].mean()}))
        print("-------------------------------------")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    q1 = dataframe[col_name].quantile(q1)
    q3 = dataframe[col_name].quantile(q3)
    interquartile_range = q3 - q1
    up_limit = q3 + 1.5 * interquartile_range
    low_limit = q1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

#Adım 6: Eksik gözlem analizi yapınız.
df.isnull().sum()
df.describe([0.01, 0.99]).T

#gebelik ve sonuç dışındaki diğer değişkenler 0 olamaz, bu değişkenlerdeki minimum değer 0 ise eksik değer vardır. hepsini tespit edip NaN yazmalıyız.
zero_col = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_col:
    df[col] = df[col].replace(0, np.nan)

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()


#Adım 7: Korelasyon analizi yapınız

df.corr()
plt.figure(figsize=(25, 20))
mask = np.triu(np.ones_like(df.corr()))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
plt.show()

corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

########################  Görev 2 : Feature Engineering

#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade
# ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
# işlemleri uygulayabilirsiniz.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)

#scale edip,  dff'e atıyoruz, imputer ile en yakın 5 komşusunu buluyoruz, ve sonrasında dff'i imputelayıp, kendi df'imize kaydediyoruz
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = imputer.fit_transform(dff)
df = pd.DataFrame(scaler.inverse_transform(dff), columns=df.columns)


#Adım 2: Yeni değişkenler oluşturunuz.
#yaş ile ilgili yeni değişkenler
df.loc[(df["Age"] >= 21) & (df["Age"] < 29), "age_interval"] = "young"
df.loc[(df["Age"] >= 29) & (df["Age"] < 41), "age_interval"] = "adult"
df.loc[(df["Age"] >= 41) & (df["Age"] <= 81), "age_interval"] = "old"

df["age_interval"] = pd.cut(df['Age'], bins=[20, 28, 40, df['Age'].max()], labels=["young", "adult", "old"])

#bmi kategorik
df["bmi_cat"] = pd.cut(x=df["BMI"], bins=(0, 18.5, 24.9, 29.9, 34.9, 39.9, 100), labels=("thin", "normal", "fat", "type 1 obese", "type 2 obese", "type 3 obese"))

#pregnancie kategorik
df.loc[(df["Pregnancies"] <= 1), "Pregnancies_num"] = "less"
df.loc[(df["Pregnancies"] > 1) & (df["Pregnancies"] <= 3), "Pregnancies_num"] = "normal"
df.loc[(df["Pregnancies"] > 3) & (df["Pregnancies"] <= 6), "Pregnancies_num"] = "high"
df.loc[(df["Pregnancies"] > 6) & (df["Pregnancies"] <= 20), "Pregnancies_num"] = "abnormal"

df["Pregnancies_num"] = pd.cut(df["Pregnancies"], bins=[0,1,3,6,20],labels=["less","normal","high","abnormal"])

#doğum yapmış mı yapmamış mı?
df.loc[(df["Pregnancies"] > 0), "Pregnancies_sit"] = "YES"
df.loc[(df["Pregnancies"] == 0), "Pregnancies_sit"] = "NO"

# Glucose seviyesi
df.loc[(df["Glucose"] < 70), "Glucose_sit"] = "low"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 110)), "Glucose_sit"] = "normal"
df.loc[(df["Glucose"] >= 110), "Glucose_sit"] = "abnormal"

df["Glucose_sit"] = pd.cut(df["Glucose"], bins=[0,69,109,df['Glucose'].max()], labels=["low","normal","abnormal"])

#vücuttaki insülin direncinin veya kan şekeri düzeylerinin belirli bir durumu gösterebilir.
df["glucose-insulin"] = df["Glucose"] * df["Insulin"]
# insülin seviyeleriyle birlikte cilt kalınlığının etkileşimi veya ilişkisi incelenebilir.
df["skinthickness-insulin"] = df["SkinThickness"] * df["Insulin"]

df.corr()


#Adım 3: Encoding işlemlerini gerçekleştiriniz.
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df

df = rare_encoder(df, 0.01)
rare_analyser(df, "Outcome", cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
#Adım 5: Model oluşturunuz.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=16)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_model, X_train)

