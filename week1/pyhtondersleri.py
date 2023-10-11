# sayılar(numbers) - Karakter dizileri(strings)
9  # integer
9.2  # float
print(9)  # print yazdırmak için kullanılır
type(9)  # tipini öğrenmek için yazılır

# atamalar ve değişkenler (assingments - Variables)
a = 9  # 9'u a'ya atıyoruz. a'yı çağırınca 9 gelir
b = "hello ai era"  # b'ye hello ai era'yı atıyoruz, çağırınca hello ai era gelir.
c = 10
a * c  # a'ya 9, c'ye 10 atadığımız için 9x10'un cevabını döndürür

d = a - c  # a ve c nin çıkarma işlemini d'ye atadık

x = 2j + 1  # complex sayı
b = 'hello ai era'  # string

True  # Boolean true/false döndürür
False
5 == 4  # bool döndürür
type(5 == 4)  # tip olarak bool der

liste = ["btc", "eth", "xrp"]  # liste
sozluk = {"name": "Peter", "age": 36}  # sözlük  key/valuedan oluşur
tupl = ("pyhton", "ml", "ds")  # tuple
se = {"pyhton", "ml", "ds"}  # set

b = 10.5  # float sayıyı
int(b)  # integera çeviriyoruz

name = "john"
name[0]  # ilk harfe erişmek için 0 yazıyoruz, pyhton dilinde sayılar 0 dan başlar
name[0:2]  # 0dan başla 2 'ye kadar git, 2 hariç

long_str = "veri yapilari asd fgh hjjk"
"veri" in long_str  # veri long_Str'nin içinde mi diye sorduk

# len fonksiyonu stringlere uygulanabilir
len(name)  # john 4 harfli, len fonksiyonuna sokunca 4 gelir

# upper() , lower() : küçük büyük dönüşümleri
"miuul".upper()
"MIUUL".lower()

hi = "hello AI Era"
hi.replace("l", "p")  # l'leri p'ye çevirdik

"hello AI era".split()  # boşluklara göre böler

" ofofof ".strip()  # boşlukları kırptı

"miuul".capitalize()  # başharfi büyütür

# listeler   değiştirilebilir /sıralı / kapsayıcı
notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "d", "v"]
not_nam = [1, 2, 3, 4, "a", "b", True, [5, 6, 7]]

not_nam[6]  # 6. elemana bakarız
not_nam[7][2]  # liste içindeki listenin içindeki elemana eriştik

not_nam[0] = 99  # ilk elemanı 99 ile değiştik

not_nam[0:4]

len(not_nam)  # içindeki sayıya bakarız
notes.append(100)  # 100 değeri listeye eklenir

notes.pop(0)  # seçilen indexdeki elemani siler
notes

notes.insert(2, 99)  # 1. argüman = index , 2. argüman girilecek olan sayı

# sözlük   /değiştirilebilir / 3.7den sonra sıralı / kapsayıcı

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and reg"}
# reg/log/cart = key
# regression/logistic reg/ classification and reg = value
dictionary.keys()
dictionary.values()
dictionary.items()  # tüm çiftleri tuple halinde listeye çevirir

# demet(tuple)    değiştirilemez/sıralı/kapsayıcı

t = ("john", "mark", 1, 2)
type(t)
t[0]
t[0:2]

# tuple'ı list(t) ile liste çevirip, istediğini ekleyip, geri tuple'a çevirirek atanır

# set  değiştirilebilir/sırasız/ kapsayıcı

set1 = {1, 3, 5}
set2 = {1, 2, 3}

set1.difference(set2)
set2.difference(set1)
set1.symmetric_difference(set2)  # 2 kümede de birbirine göre olmayanlar

set1.intersection(set2)  # iki kümenin kesişimi

set1.union(set2)  # iki kümenin birleşimi

set1.isdisjoint(set2)  # iki kümenin kesişimi boş mu?

set1.issubset(set2)  # bir küme diğerinin alt kümesi mi?

set1.issuperset(set2)  # bir küme diğerini kapsıyor mu ?


########## FONKSIYONLAR####################

def calculate(x):
    print(x * 2)


calculate(5)


def summer(arg1, arg2):
    print(arg1 + arg2)


summer(80, 24)


# DocString : fonksiyonlarımıza eklediğimiz bilgi notu, başka birisi kullanmak isterse diye

# fonksiyonların statement bölümü

# def funchtion(paramaters/arguments):
#      statements(function body)


def say_hi():
    print("Merhaba")
    print("hi")
    print("Hello")


say_hi()


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# girilen değerleri bir liste içinde saklayacak fonksiyon

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 8)
add_element(180, 10)


def divide(a, b=5):  # b'ye 5 atadık ön tanımlı, fonksiyonu yazarken 2. argüman girmeye gerek yok
    print(a / b)


divide(8)


# kendini tekrar etmesi gereken zamanlarda fonksiyon yazmamız lazım, tek tek uğraşmamak için


def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78)

## döngüler

if 1 == 1:
    print("doğru")

if 1 == 2:  # eşit olmadığı için boş döndürdü
    print("yanlış")


def number_check(number):
    if number == 10:  # eğer numara 10 ise
        print("number is 10")  # bu kısmı döndürür
    else:  # eğer numara 10 değil ise
        print("number is not 10")  # bu kısmı döndürür


number_check(10)
number_check(12)


def numbers(number):
    if number > 10:  # eğer numara 10'dan büyükse
        print("greater than 10")  # bunu döndürür
    elif number < 10:  # eğer numara 10'dan küçükse
        print("less than 10")  # bunu döndürür
    else:  # eğer yukardaki iki koşulu karşlamıyosa
        print("equal to 10")  # ki bu durum eşit olması, bunu döndürür


numbers(10)
numbers(11)
numbers(9)

# döngüler (loops)

students = ["john", "mark", "venessa", "mariam"]

for i in students:  # students listesindeki hepsini yazdırmak için
    print(i)

for student in students:  # fonksiyonu tanımladık
    print(student.upper())  # fonksiyonun içinde olucak olay, listedekileri büyült

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:  # maaşla hakkında loop yazdık
    print(int(salary * 20 / 100 + salary))  # her maaşa %50 zam


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1500, 10)

salaries2 = [10700, 2500, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


def alternating(string):  # fonksiyonu tanımlıyoruz
    new_string = ""  # girilecek string
    for string_index in range(len(string)):  # stringin harflerini gezecek döngü
        if string_index % 2 == 0:  # eğer 2'ye bölümü kalan 0 ise
            new_string += string[string_index].upper()  # harfi büyült
        else:  # eğer 2'ye bölümü 0 değil ise
            new_string += string[string_index].lower()  # harfi küçült
    print(new_string)  # stringi yazdır


alternating("Hello world i am learning pyhton")

# break = loopu durdurur
# continue = atadığımız continue değerini görünce o değeri atlar
# while = 'dı sürece demektir, atanılan değere while ile verilen değere gelene kadar denenir
number = 1
while number < 5:  # 1-2-3-4 döndürür, 5 görünce durur
    print(number)
    number += 1

# enumerate : otomatik counter ile for loop

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []
for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

# mülakat sorusu
students = ["John", "Mark", "Venessa", "Mariam"]  # öğrenci listesi


def divide_students(students):  # fonksiyonu tanımladık
    groups = [[], []]  # 2 farklı grup açtık
    for index, student in enumerate(students):  # studentsin içinde indexleri tek tek gez
        if index % 2 == 0:  # eğer kalanı 0 ediyosa
            groups[0].append(student)  # 1. listeye ekle
        else:  # eğer kalanı 0'dan farklıysa
            groups[1].append(student)  # 2. listeye ekle
    print(groups)  # listeleri yazdır
    return groups  # listeleri döndür


divide_students(students)


# mülakat sorusu
def alternating_with_enumerate(string):  # fonksiyonu kuruyoruz
    new_string = ''  # geçici boş  stringi tanımladık / for döngüsünün çıktılarını kaydetmek için
    for i, letter in enumerate(string):  # stringin içinde index index gez
        if i % 2 == 0:  # eğer kalanı 0 ise
            new_string += letter.upper()  # harfi büyült
        else:  # eğer kalanı 0'dan farklı ise
            new_string += letter.lower()  # harfi küçült
    print(new_string)  # stringi yazdır


alternating_with_enumerate("hi my name is john and i am learning pyhton")

# zip / farklı listeleri aynı listede birleştirdik

students = ["John", "Mark", "Venessa", "Mariam"]
departments = ["mathematics", "statistic", "physics", "astronomy"]
ages = [23, 30, 25, 22]
list(zip(students, departments, ages))

# lambda , map , filter ,reduce

new_sum = lambda a, b: a + b
new_sum(5, 7)

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

list(map(lambda x: x * 50 / 100 + x, salaries))

# comprehensions
# list comprehension

salaries = [1000, 2000, 3000, 4000, 5000]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

students = ["john", "mark", "venessa", "mariam"]
students_no = ["john", "venessa"]
# studentsin içinde tek tek gez, eğer students_no'dan eleman studentsin içindeyse onun adını küçük yaz, değilse büyük yaz
[student.lower() if student in students_no else student.upper() for student in students]

# dict comprehension

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}
# key değerleri sabit kalıp, value değerlerin karesini al
{k: v ** 2 for (k, v) in dictionary.items()}
# key değerleri büyük harfle yazıp, valuelar sabit kalsın
{k.upper(): v for (k, v) in dictionary.items()}

# mülakat sorusu
# amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir.
# keyler orjinal değerler, valuelar ise değiştirilmiş olacak

numbers = range(10)  # 0dan 10 a kadar sayı oluşturduk
new_dict = {}  # yeni sözlük oluşturduk

# key sabit/value'nun karesi #sayılarda gez # eğer kalan 0 ise
{n: n ** 2 for n in numbers if n % 2 == 0}

# list/ dict comprehension uygulamalar

# bir veri setindeki değişken isimlerini değiştirmek

# before : ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses', 'abbrev']

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() for col in df.columns]
df.columns

# isminde "INS" olan değişkenlerin başına flag diğerlerine NO_FLAG eklemek istiyoruz.

[col for col in df.columns if 'INS' in col]
#kolon isimlerinde gez, INS ile başlıyanların başına FLAG ekle
["FLAG_" + col for col in df.columns if "INS" in col]
#kolon isimlerinde gez, INS ile başlıyanların başına FLAG ekle, başlamıyanların başına NO_FLAG ekle
["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

#mülakat sorusu
# Amaç key'i string, value'su mean,min,max,var olan liste oluşturmak
#sadece sayısal değişkenleri yapmak istiyoruz
df = sns.load_dataset("car_crashes")
df.columns
#herhangi bir methoda gerek kalmadan list of comprehension ile, kolonlarda gezip, dtype'ı object olmayanları aldık
num_cols = [col for col in df.columns if df[col].dtype != "O"]
num_cols

soz = {}
agg_list = ["mean", "min", "max","sum"]

for col in num_cols:
    soz[col] = agg_list

#kısa yol

new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

df[num_cols].head()



