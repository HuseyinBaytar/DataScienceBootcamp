## matplotlib

#kategorik değişken: sütun grafik. countplot bar
#sayısal değişken: histogram, boxplot

#kategorik değişken görselleştirme

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("titanic")

df["sex"].value_counts().plot(kind="bar")
plt.show(block=True)


# sayısal değişken görselleştirme

plt.hist(df["age"])
plt.show(block=True)

plt.boxplot(df["fare"])
plt.show(block=True)


#matplotlib'in özellikleri
import numpy as np
#plot: veriyi görselleştirmek için kullandığımız fonksiyon
x= np.array([1,8])
y = np.array([0,150])

plt.plot(x,y)
plt.show(block=True)

# marker : işaretleyici özellikleri, marker = 'o'

#line : çizgi özelliği, linestyle="dashed" / "dotted" / "dashdot" ,  color= "renk baş harfi"

#multiple lines: önce x sonra y'yi yazdırınca üstüne yazdırır

######labels#####
#başlık için plt.title("")
#x ekseni  plt.xlabel(")
#y ekseni  plt. ylabel(")
#ızgara arkaplanda,  plt.grid()


## subplots ###
#1 satırlık, 2 sütünluk, 1. grafiği çıkarır
#plt.subplot(1,2,1)


################# SEABORN #############################
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

#kategorik
# df[sex].value_counts()
#sns.counplot(x=df["sex"],data=df)
#plt.show()

#sayısal
#sns.boxplot(x=df["total_bill"])
#plt.show()

#df["total_bill"].hist()
#plt.show()








