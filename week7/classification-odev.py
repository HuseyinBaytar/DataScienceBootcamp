##################### Sınıflandırma Modeli Değerlendirme

#Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma modeli oluşturulmuştur. 10 test verisi gözleminin gerçek değerleri ve modelin tahmin ettiği olasılık değerleri verilmiştir.

import pandas as pd

#- Eşik değerini 0.5 alarak confusion matrix oluşturunuz
#- Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.

gercek_deger = [1,1,1,1,1,1,0,0,0,0]
probability = [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]

dict = {'gercek': gercek_deger, 'probability': probability}

df = pd.DataFrame(dict)

from sklearn.metrics import confusion_matrix

tahmin = [1 if p > 0.5 else 0 for p in probability]
def plot_confusion_matrix(y, y_pred):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y,y_pred,labels=[1,0])
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xticks([0.5,1.5],[1,0])
    plt.yticks([0.5, 1.5], [1, 0])
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(df['gercek'], tahmin)

#True positive  / False negative
#False positive  / True negative

#1 ken 1
TP = 4
#0 ken 1
FP = 1
#1 ken 0
FN = 2
#0 ken 0
TN = 3

# Accuracy hesaplama:  Doğru sınıflandırma oranı (tp+tn) / (tp+tn+fp+fn)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Recall hesaplama : Pozitif sınıfın doğru tahmin edilme oranıdır  TP / (TP + FN)
recall = TP / (TP + FN)

# Precision hesaplama : Pozitif sınıf tahminlerinin başarı oranı TP / ( TP + FP )
precision = TP / (TP + FP)

# F1 Score hesaplama : 2 * (precision * Recall) / ( Precision + Recall)
f1_score = 2 * (precision * recall) / (precision + recall)


# Görev 2 : Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur. %90.5 doğruluk oranı elde edilen modelin başarısı
# yeterli bulunup model canlıya alınmıştır. Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi olmamış. iş birimi modelin başarısız olduğunu iletmiştir.
# Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir.

# Buna göre;
# -Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.
# -Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.

TP = 5
FP = 90
FN = 5
TN = 900

# Accuracy hesaplama:  Doğru sınıflandırma oranı (tp+tn) / (tp+tn+fp+fn)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Recall hesaplama : Pozitif sınıfın doğru tahmin edilme oranıdır  TP / (TP + FN)
recall = TP / (TP + FN)

# Precision hesaplama : Pozitif sınıf tahminlerinin başarı oranı TP / ( TP + FP )
precision = TP / (TP + FP)

# F1 Score hesaplama : 2 * (precision * Recall) / ( Precision + Recall)
f1_score = 2 * (precision * recall) / (precision + recall)

#FN değerinin modelin dolandırıcılık durumlarını kaçırdığı ve gerçek dolandırıcık vakalarını tespit etmede zayıf olduğunu gösterir. Kullancılardan gerçekte dolandırıcı olmayan insanları dolandırıcı diye suçluyor. Bu durum haksız suçlamalarla, kullanıcının güvenini kaybetmeye neden olabilir. şirketin veri bilimi takımı FN'yi düşürmeye yönelik bir çalışma yapmalıdır.
# daha fazla veri topluyabilir, model parametrelerini ayarlıyabilir, farklı model denemeleri yapabilir.

