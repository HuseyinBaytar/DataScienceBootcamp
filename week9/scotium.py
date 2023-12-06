##########   Scoutium Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma

#İş Problemi

#Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf(average, highlighted) oyuncu olduğunu tahminleme.

#Veri Seti Hikayesi

#Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını
# içeren bilgilerden oluşmaktadır.

######## Değişkenler
# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id :Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# position_id : İlgili oyuncunun o maçta oynadığı pozisyonun id’si
# 1: Kaleci
# 2: Stoper
# 3: Sağ bek
# 4: Sol bek
# 5: Defansif orta saha
# 6: Merkez orta saha
# 7: Sağ kanat
# 8: Sol kanat
# 9: Ofansif orta saha
# 10: Forvet
# analysis_id : Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier , GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)

#Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

attributes_df = pd.read_csv("week9/datasets/scoutium_attributes.csv", sep=";")
potential_labels_df = pd.read_csv("week9/datasets/scoutium_potential_labels.csv",sep=";")

attributes_df.head()
potential_labels_df.head()


#Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden
# birleştirme işlemini gerçekleştiriniz.)

df = attributes_df.merge(potential_labels_df, how="left", on=["task_response_id", "match_id" , "evaluator_id","player_id"])

df.head()
df.shape
df.info()
df.isnull().sum()

#Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df["position_id"].value_counts()

df[df["position_id"] != 1]

df = df[df["position_id"] != 1]

#Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df.shape

df[df['potential_label'] != "below_average"]

df = df[df['potential_label'] != "below_average"]

# Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
#indekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan“attribute_value” olacak şekilde pivot table’ı oluşturunuz.  “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz


pivot_table_df = pd.pivot_table(df, values="attribute_value", index=["player_id","position_id","potential_label"], columns="attribute_id").reset_index()
pivot_table_df.head()
pivot_table_df.columns

pivot_table_df.columns = [str(col) for col in pivot_table_df.columns]


#Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz

pivot_table_df["potential_label"].value_counts()

pivot_table_df["potential_label"] = pivot_table_df["potential_label"].map({"average":0,"highlighted":1})


#Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
pivot_table_df.dtypes

num_cols = [col for col in pivot_table_df.columns if pivot_table_df[col].dtype == "float64"]

num_cols

#Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız

pivot_table_df[num_cols] = StandardScaler().fit_transform(pivot_table_df[num_cols])

pivot_table_df.head()


#Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
#geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
X = pivot_table_df[num_cols]
y = pivot_table_df["potential_label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

def base_models(X, y, rs, scoring):
    print("Base Models....")
    y = y.values.ravel()
    classifiers = [
        ('LR', LogisticRegression(random_state=rs)),
        ("CART", DecisionTreeClassifier(random_state=rs)),
        ("RF", RandomForestClassifier(random_state=rs)),
        ('XGBoost', XGBClassifier(verbose=-1 , use_label_encoder=False, eval_metric='logloss', random_state=rs)),
        ('LightGBM', LGBMClassifier(verbose=-1, random_state=rs)),
        ('CatBoost', CatBoostClassifier(logging_level = "Silent",random_state=rs))
    ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for i in scores:
    base_models(X, y, 1, i)


#Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

def feature_importance(model, X_train, y_train):
    model.fit(X_train, y_train)
    feature_importance = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["importance"])
    feature_importance.sort_values(by="importance", ascending=False, inplace=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance.importance[:10], y=feature_importance.index[:10], palette="rocket")
    plt.title("Feature Importance")
    plt.show()

feature_importance(CatBoostClassifier(logging_level = "Silent",random_state=1), X_train, y_train)

