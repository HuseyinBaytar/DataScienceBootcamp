# Advanced Functional EDA

#1- Genel resim
#head,tail,shape,info,columns,index,describe,isnull.values.any , isnull.sum

# def check_df(dataframe, head=5):
#             print("################# shape ###############")
#             print(dataframe.shape)
#             print("################ types ################")
#             print(dataframe.dtypes)
#             print("################ Head ################")
#             print(dataframe.head(head))
#             print("################ Tail ################")
#             print(dataframe.tail(head))
#             print("################ Quantiles ################")
#             print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)

#check_df(df)



#2- Kategorik değişken analizi
#tüm columnlarda gez, eğer tipi category, objct veya bool ise yakala, cat_Cols'un içine ata
# cat_cols = [ col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]

#tüm columnlarda gez eğer içindeki değişkenler 10'dan az ise ve tipi integer/float ise, numeric but categoric'e ata
#numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]

#kardinaller = bir kategorik değişkenin çok fazla eşsiz değere sahipse o kardinaldir, örneğin: isim değişkeni hepsi unique bi değer taşımıyor
#categoric_but_cardinal = [ col for col in df.columns if  df[col].nunique > 20 and str(df[col].dtypes) in ["category","object"]

#cat_cols = cat_cols + categoric_but_cardinal      # tüm kategorikleri bi yere atarız

# cat_cols = [col for col in cat_cols if col not in cat_but_car]
# df[cat_cols].nunique()

# def cat_summary(dataframe, col_name):
#           print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                               "Ratio": 100 * dataframe[col_name].value_counts() / len/dataframe)}))

#tüm kategorik değişkenlerde gez ve yukardaki fonksiyonu yazdır
#for col in cat_cols:
#     cat_summary(df,col)



# def cat_summary(dataframe, col_name, plot=False):
#           print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                               "Ratio": 100 * dataframe[col_name].value_counts() / len/dataframe)}))
#           print("################################")
#           if plot:
#                sns.countplot(x=dataframe[col_name], data=dataframe)
#                plot.show(block=True)

#cat_summary(df,colname, plot=True)

#tek tek kategoriklerde gez ve görselleştir/summarysini ver
#for col in cat_cols:
#     if df[col].dtypes == "bool":
#        df[col] = df[col].astype(int)
#        cat_summary(df,col, plot=True)
#     else:
#       cat_summary(df,col, plot=True)



#3- Sayısal değişken analizi

# df["numeric"].describe().T
#numericleri yakalamak // ama numeric but catlar gelebilir
#[col for col in df.columns if df[col].dtypes in ["int","float"]

#cat_colsta olmayan numeric cols'u yakaladık
#num_cols = [col for col in num_cols if col not in cat_cols]

# def num_summary(dataframe,numeric_col, plot=False):
#          quantiles = [0.05, 0.20, 0.30, 0.50, 0.75, 0.99]
#          print(dataframe[numerical.col].describe(quantiles).T)
#          if plot:
#                  dataframe[numerical_col].hist()
#                  plt.xlabel(numeric_col)
#                  plt.tittle(numeric_col)
#                  plt.show(block=True)



#tüm numerik collarda gez %'lik bilgileri yakala
# for col in num_cols:
#       num_summary(df, col, plot=True)



#4- Hedef değişken analizi

#def grab_col_names(dataframe, cat_th=10, car_th=20):
#   """ doc string """
#   cat_cols = [ col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]
#   numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]
#   categoric_but_cardinal = [ col for col in df.columns if  df[col].nunique > 20 and str(df[col].dtypes) in ["category","object"]
#   cat_cols = cat_cols + categoric_but_cardinal
#   cat_cols = [col for col in cat_cols if col not in categoric_but_cardinal]

#   num_cols =[col for col in df.columns if df[col].dtypes in ["int","float"]
#   num_cols =[col for col in num_cols if col not in cat_cols]

#   print(f"Observations: {dataframe.shape[0]")
#   print(f"Variables: {dataframe.shape[1]")
#   print(f"cat_cols: {len(cat_cols)}")
#   print(f"num_cols: {len(num_cols)}")
#   print(f"categoric_but_cardinal: {len(categoric_but_cardinal)}")
#   print(f"numeric_but_categoric: {len(numeric_but_categoric)}")

#   return cat_cols, num_cols, categoric_but_cardinal

# cat_cols, num_cols, categoric_but_cardinal = grab_colnames(df)



#hedef değişken = odaklandığımız değişken örneğin, survived,churn ,araç fiyat vs

#df["hedef"].value_counts()
#değişkenleri hedefe göre çaprazlamamız lazım

#df.groupby("kategorik-değişken")["target"].mean()    = hedef değişken ile seçtiimiz kategorik değişken arasındaki ortalamayı alırız

#fonksiyon ile ortalamasına bakmayı yazdık
# def target_summary_with_cat(dataframe, target, categorical_col):
#           print(pd.DataFrame({"Target_mean": dataframe.groupby(categorical_col)[target].mean()

#tek tek tüm hepsini gezdiriyoruz
# for col in cat_cols:
#       target_summary_with_cat(df, "survived", col)


##hedef değişkeni sayılsa değişkenler ile analizlemek

#fonksiyon ile target ile numeric arasındaki ortalamaya bakarız
# def target_summary_with_num(dataframe, target, numerical_cols):
#       print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

#tek tek gezer
#for col in num_cols:
#   target_summary_with_num(df, "survived", col)



#5- Korelasyon analizi
#numerik columsnlari yakaladık
# num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
#corelasyonunu çıkardık, eğer ilişki 1'e yakınsa pozitif denir, -1'e yakında negafif denir. 0 civarı = korelasyon yok
# corr = df[num_cols].corr()

# sns.heatmap(corr, cmap="RdBu")
#plt.show()


#yüksek korelasyonlu değikenlerin silinmesi
# çok yüksek korelasyonluysa eğer, aynı gibidir 2 değişken, o yüzden 1'i silinebilir
#cor_matrix = df.corr().abs()  #mutlak değerini aldık
#upper_triangle_matrix=cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))


#drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]<0.90)]

#df.drop(drop_list, axis=1)  droplar


#def high_corelated_cols(dataframe,plot=False, corr_th= 0.90):
#       corr= dataframe.corr()
#       corr_matrix = corr.abs()
#       upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
#       drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]<0.90)]
# if plot:
#       import seaborn as sns
#       import matplotlib.pyplot as plt
#       sns.set(rc={'figure.figsize':(15,15)})
#       sns.heatmap(corr, cmap="RdBu")
#       plt.show()
#   return drop_list

#high_corelated_cols(df)


