############################Regresyon Modelleri için Hata Değerlendirme
#Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.

import pandas as pd
import numpy as np

deneyim_yili = [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1]
maas = [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]

dict = {'Deneyim Yılı': deneyim_yili, 'Maaş': maas}

df = pd.DataFrame(dict)

df

# 1-Verilen bias ve  Weight’e göre doğrusal regresyon model denklemini oluşturunuz.
# Bias=275,Weight=90 (y’=b+wx)

#y_hat = b + w*x

# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
275 + 90 * 5
275 + 90 * 7
275 + 90 * 3
275 + 90 * 2
275 + 90 * 10
275 + 90 * 6
275 + 90 * 4
275 + 90 * 8
275 + 90 * 1
275 + 90 * 9

df["Predict"] = 275 + 90 * df["Deneyim Yılı"]

df

# 3-Modelin başarısını ölçmek için MSE,RMSE,MAE skorlarını hesaplayınız.

#maaş - maaş tahmin = hata,  hata'nın karelerinin toplamının ortalaması = MSE

df["hata"] = df["Maaş"] - df["Predict"]

df["hata_kareleri"] = [hata ** 2 for hata in df["hata"]]

toplam = sum(df["hata_kareleri"])

MSE = toplam / 15
print(MSE)


#RMSE: MSE'dekinin karekök alınmış halidir.
rmse = np.sqrt(MSE)

MSE ** (1/2)

print(rmse)

df
# MAE : gerçek değerler ile tahmin edilen değerler arasındaki mutlak farkların ortalamasını ifade eder.

df["mutlak_hata"] = [abs(hata) for hata in df["hata"]]

MAE = df['mutlak_hata'].sum() / len(df)
print(MAE)
