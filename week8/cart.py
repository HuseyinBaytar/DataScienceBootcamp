# classification and regression Tree  ( CART )
#cart 1982 yılında Leo Breiman tarafından ortaya atılmış bir yöntemdir. Random forest ve diğer kullanılan bir çok yöntemin temelini oluşturur.
#amaç veri seti içerisindeki karmaşık yapıları basit karar yapılarına dönüştürmektir. Heterojen veri setleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır.

# Kural üzerinden gider, örneğin bir kişinin deneyim yılı 4 yıldan büyükse 520 maaşın üstü , 4 yıldan küçükse  520 maaşın altı. Bunlarda kendi içinde farklı kırılımlar yapabilir örneğin, 4 yıldan büyük ve maaşı 520'in üstü ama dil bilme durumunda bir kırılım olabilir yani, 520 maaş bildiği dil sayısı 3 den fazlaysa maaş 800'ün üstü, bildiği dil sayısı 3 den azsa 600'ün altı gibi.

#bağımsızları bölen noktalara iç düğüm noktaları denir. mesela 4 yildan büyük veya küçükün olduğu ayrım.  internal noddlar 2 tane, terminal nodelar(yaprakların sonları 4 tane) vardır. yukardaki örnekte. örneğin internal nodelar = deneyim yılı ayrımı 4 yıl ve dil bilme sayısı 3 dil, terminal noddlar 4 tane = 520 maaş, 600 maaş ve 800 maaş.

#regresyon problemi karar ağaç yapısı,
#if predictor A >= 1.7 then
#   if Predictor B >= 202.1 then outcome = 1.3
#   else outcome = 5.6
#else outcome = 2.5

# bir karar ağacı kullandığımızda, bize bir karar kuralları çıkarır, örneğin predictor A eğer 1.7 den büyük yada eşitse, tahminci B 202.1den büyük yada eşitse çıktı 1.3'dür demiş, eğer değilse 5.6'dır demiş, ve eğer tahminci A eğer 1.7'den küçükse 2.5'dir demiştir.

# Eğer bu kadar kolaysa niye bunu excelde,pyhtonda,sql'de yapmıyoruz?
# Bunun algoritmik bir şekilde belirli referans noktalarına göre hareket ederken kendimizi optimize ederek bu karar kurallarının ne olduğunu belirlemek. Otomatik olarak 1.7 den böldük tamamda neden 1.7'den böldük? Bu alt grupları ki-kare,Gini, Entropi, SSE gibi bazı karar kuralı metrikleriyle bölüyor olucaz. Algoritmalarımız buna göre bölüyor.

#Regresyon problemleri için Cost fonksiyonu
#RSS fonksiyonu.
#bir ağacı nasıl oluşturmalıyım sorusunun yanıtı;
#Rss/SSE değerinin minimum olduğu noktalardaki bölgeler, yapraklar, kutular bölünmesi gereken yerdir. Gerçek değerler Yi, yaninda yazan Rj bize bölgeler/yapraklar/kutuları gösterir. Her açıdan bölünür ve bölüm noktalarındaki en küçük Hatalar yani SSE değerlerinin olduğu yer kabul edilir ve birinci dallanma tamamlanır. 2 ayrı dala ayrıldıktan sonra bu dalları ayrı ayrı tüm veri gibi kabul edilip dallanma işlemlerini tekrar eder.
#Kaç kere bölme işlemi yaptığını model bilmiyor, o yüzden parametrelerinde bizim vermemiz lazım. Örneğin bir dalda son 2 değer kaldı, eğer onu bölerse çok iyi öğrenir overfittinge gidebilir. eğer bölmezse rassallığı korumuş olur.  Max depht ve min samples split argümanları bizim için çok önemlidir.

#Sınıflandırma problemleri için Cost fonksiyonu
#Gini ve Entropiyi kullanabiiriz. Elimizde gerçek sınıflar var, Tahmin ettiğimiz sınıflar var bunların arasındaki durumu değerlendirerek bize başarımıza ilişkin bir bilgi verir. bu başarı metriklerine göre karar verilir. Saflık ölçüleri denir. Ne kadar düşükse o kadar iyidir.
#Entropi bir çok disiplinde kullanılan bir değerdir. bizim ilgilendiğimiz kısmı 'ne kadar düşük entropi o kadar iyidir'. Tahmin ettiğimiz değerlerin çeşitli olmamasını ve düşük olmasını bekleriz.
#Bir daldayız, çeşitli tahminlerimiz var ve 2 sınıfımız olduğunu düşünelim, bir sınıfımızın gerçekleşme olasılığı çarpı logaritmasıdır. Bu değerler ne kadar az çeşitli olursa o kadar iyi olur.

#Gini, K1'den tüm sınıf sayısı kadar, mesela 0'ıncı sınıfın gerçekleşme olasılığı çarpı gerçekleşmeme olasılığı +  1. sınıfın gerçekleşme olasılığı çarpı gerçekleşmeme olasılığı, Yani aslında burda entropiye benzer bir durum var, bir sınıfın gerçekleşme olasılığı ne kadar yüksekse, diğer sınıfın gerçekleşme olasılığı o kadar düşüktür.


################################################
# Decision Tree Classification: CART
################################################



# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
#  Modeling using CART
################################################
#dataseti okutuyoruz
df = pd.read_csv("week7/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#modelimizi kuruyoruz
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)

#%100 tahmin ediyoruz, Teorik olarak mümkün değil, raslantısal bir hata olması lazım. Başarı nasıl %100 çıkabilir? overfit ? successfull?
#Lets test it with different methods


#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################
#veri setini 2'ye ayırıyorum %70'e %30 olarak.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#trainde hatamız yok görüyoruz.

# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#modelimizin hiç görmediği test setini gönderdik ve görüyoruz ki, model eğitildiği veride performasını çok yüksek gösterdi ama görmediği bir veriyi sorduğumuzda nerdeyse yarı yarıya fark etti.
#overfit olduğu için hiç görmediği bir veriyi sorduğumuda başarısı düştü.


#####################
# CV ile Başarı Değerlendirme
#####################
#modelimizi kuruyoruz
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
#cross validate ile başarılarına bakıcaz
cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

#Tüm veriyi modellemek olarak test ettiğimizde overfit oldu. Daha sonrasında train ve test ile ayırıp baktık, train seti yine 1 çıktı ama test hatamız beklediğimiz gibi düştü. Train seti ile TEst setimizin rastgeleliğini değiştirdik ve bu sefer daha iyi sonuçlar aldık

################################################
# Hyperparameter Optimization with GridSearchCV
################################################
#bizim için önemli olanlar: min_sample_split mesela ön tanımlı değeri 2, 2 tane görünce bölüyor yani overfit olabilir. Max_depth bizi ilgilendiren başka bir parametre. overfittin önüne geçebilicek parametreler
cart_model.get_params()
#max depthi 1 ile 11e kadar, sample'ı 2-20'ye kadar aralıklar alıyoruz
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

#Eğer roc auc score bakmak istersek, parametre olarak Auc score verilebilir.
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
#max depth 5, min sample 4'ün en iyi olduğunu söylüyor
cart_best_grid.best_params_
#en iyi score olarakta 0.75 almış.
cart_best_grid.best_score_

#random bir sample alıp deniyoruz
random = X.sample(1, random_state=45)
cart_best_grid.predict(random)


################################################
# Final Model
################################################
#en iyi değerlerle final modeli kuruyoruz
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()
#min sample split 4, max depth 5

#aynı şekilde, set params şekli ile daha önceden kurduğumuz modelin üzerinde set eder
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
#5 katlı şekilde bakıyoruz final modelimize
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.75
cv_results['test_f1'].mean()
#0.61
cv_results['test_roc_auc'].mean()
#0.79
#başarılı bir şekilde hipermetre optimizasyonunu tamamladık

################################################
# Feature Importance
################################################
#hangi Featureların en çok etkili olduklarını görselleştirerek görüyoruz. Amaçlarımıza en çok hizmet eden değişken glucouse
cart_final.feature_importances_

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


plot_importance(cart_final, X, num=5)

################################################
#Analyzing Model Complexity with Learning Curves (BONUS)
################################################

train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc",
                                           cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

#maximum depth para metresinin hem test hemde train içinde 1 ile 11 arasında deneme yapıyoruz, en iyi değerlerini alıyoruz. Görselleştirerek bunu daha rahat anlayabiliriz.
#biz zaten hiperparemetrelere bakıp seçebiliceğimiz en iyisini seçtik, 4 seçmiştik ama sanki burda 3'ü mü seçmeliyiz? burda baktığımız şey sadece max depht, belki max depth'i 3 seçersek, diğer paremetlereler bozulabilir.

plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')
plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()



#yukarda yaptığımız işlemleri fonksiyonlaştırdık
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

#ilgili hiperparemetrenin ilgili öğrenme aralığını gösterdi
val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

#mesela burda ayrı ayrı 2 parametreye bakıyoruz
cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])



################################################
# Visualizing the Decision Tree
################################################
# conda install graphviz
# import graphviz

#çalışmanın başındaki model'i görselleştirecek bir fonksiyon
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

#çalıştırarak resmimizi elde ediyoruz
tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")


################################################
# Extracting Decision Rules
################################################
#we can see our decision rules on console
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)


################################################
# Extracting Python Codes of Decision Rules
################################################
#in this section, we can get the pyhton, sql and excel codes with thoose codes.

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))


################################################
# Prediction using Python Codes
################################################

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)


#columnslarımıza bakıyoruz
X.columns
#rastgele değer atıyoruz
x = [12, 13, 20, 23, 4, 55, 12, 7]
#tahmin ediyoruz
predict_with_rules(x)
#rastgele değer atıyoruz
x = [6, 148, 70, 35, 0, 30, 0.62, 50]
#tahmin ediyoruz
predict_with_rules(x)

################################################
# Saving and Loading Model
################################################
#kurulan modeli pkl dosyası olarak çıkarıyoruz, çalıştığımız şirkettekilere vs yollamak için
joblib.dump(cart_final, "cart_final.pkl")
#dışarıdan bir modeli yüklüyoruz
cart_model_from_disc = joblib.load("cart_final.pkl")

#random değerler atıyoruz
x = [12, 13, 20, 23, 4, 55, 12, 7]
#tahmin ediyoruz
cart_model_from_disc.predict(pd.DataFrame(x).T)

