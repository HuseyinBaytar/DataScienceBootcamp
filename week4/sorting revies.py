#iyi yorum veya kötü yorum farketmez, diğer insanların o yorumu faydalı bulması öne çıkarmalıdır.

# üst- alt farkı skoru

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)


# up- down diff score =  up ratings -  down ratings

#review 1: 600 up 400 down total 1000
#review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up,down):
    return up - down

#review1  cevap 200
score_up_down_diff(600,400)
#%60 pozitif score

#yüzde ola

#review2  cevap 1000  review2 kazanıyo gibi gözüksede yüzdelikleri farklıdır,1. kazanır çünkü total vote'un değeri
score_up_down_diff(5500,4500)
#%55 pozitif score





###########################
# Ortalama puanı (Average RAting)   score  =  up ratings / all ratings


def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up+down)

score_average_rating(600,400)
#0.6 döndürdü

score_average_rating(5500, 4500)
#0.55 döndürdü
#faydalı olanı olarak isimlendirilebilir.


#review1: 2 up 0 down total 2
#review2: 100 up 1 down total 101

score_average_rating(2,0)
#1 verdi

score_average_rating(100, 1)
#0.99 verdi

#frekans yüksekliğini sayı yüksekliğini göz önünde bulduramadı.



###########################
# Wilson lower bound score

#bize 2li interactionlar barından herhangi bir item,product,reviewi skorlama imkanı sağlar
#örn yorumlar like/dislike, youtube videolar like/dislike etc..

#bernolli

def wilson_lower_bound(up,down,confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1- (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n )) / (1 + z * z / n)


wilson_lower_bound(600,400)

wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)

wilson_lower_bound(100,1)


###############################################
#Case Study
######################################
up = [15,70,14,4,2, 5,8, 37, 21 ,52, 28, 147, 61, 30, 23, 40,37,61]
down = [0,2,2,2,15,2,6,5,23,8,12,2,1,1,5,1,2,6]

comments = pd.DataFrame({"up": up, "down": down})

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]),
                                                axis=1)


comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)


comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)


comments.sort_values("wilson_lower_bound", ascending=False)




#önümüzde bir puan hesabı işi olduğununda;  bi average alabilirim ve bunu hassaslaştırabilirim, örneğin zamana dayalı veya kullanıcı davranışlarına dayalı diğer yandan, elimde 5 yıldızlı bir rating olduğunda bunu bayesian average rating ile hesaplıyabilrim, bunların hepsi ayrı ayrı kullanılıcak yöntemler bunların hepsini bir kerede hybird ile kullanabilirim.

#sıralamada yine göz önünde birden fazla faktör olduğu bilgisi var, bu faktörlere ağırlık vermem lazım,  ilgili probleme özel kişisel matematik ilede yapılabilir bi sistemde olabilir like imdb,  en genelinde baktığında alt birimimiz puanlamak, puanlamaya göre sıralıyoruz, bayesiyan rating score sistemini öğrendik, bu yöntem puanların dağılım bilgisine göre bize bir olasılıksal ortalama hesabı yapıyodu, bu ortalamayı nihai ortalama olarak kullanabiliriz ama biraz puanları kırpabilir o yüzden ekstradan faktörler ile hybrid şekilde puan dağılımları üzerinden bar skorunu, satın alma sayılarını, normal raitingleri ve yorum sayılarını göz önünde bulundurarak hybrid bir yol yaptık.








