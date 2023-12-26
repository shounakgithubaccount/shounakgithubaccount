- 👋 Hi, I’m @shounakgithubaccount
- 👀 I’m interested in data analysis , cleaning , updating and building new datasets using data wrangling techniques 
- 🌱 I’m currently learning Python and its application in ML , AI using algorithms 
- 💞️ I’m looking to collaborate on  working in Real time based project 
- 📫 How to reach me ... - - through my linkedin - shounsak.mitra/in 

<!---
shounakgithubaccount/shounakgithubaccount is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importiong the data
DF=pd.read_csv('CapstoneDataSet.csv')  
#Checking the values of our given dataset
DF.head()
   Cust_Id  Rating  Movie_Id   Genre        MovieName
0  1488844       3         1  Action  Dinosaur Planet
1   822109       5         1  Action  Dinosaur Planet
2   885013       4         1  Action  Dinosaur Planet
3    30878       4         1  Action  Dinosaur Planet
4   823519       3         1  Action  Dinosaur Planet
#df2 = DF.drop_duplicates(subset=["Movie_Id", "Genre","MovieName"], keep='first')


#DF.drop(DF(DF['Rating'] >= 4).index, inplace=False)

#DF

#df2 = DF.drop(DF[(DF['Rating'] <= 4)].index, inplace=False)
DF.shape
(1048574, 5)
DF.head(10)
   Cust_Id  Rating  Movie_Id   Genre        MovieName
0  1488844       3         1  Action  Dinosaur Planet
1   822109       5         1  Action  Dinosaur Planet
2   885013       4         1  Action  Dinosaur Planet
3    30878       4         1  Action  Dinosaur Planet
4   823519       3         1  Action  Dinosaur Planet
5   893988       3         1  Action  Dinosaur Planet
6   124105       4         1  Action  Dinosaur Planet
7  1248029       3         1  Action  Dinosaur Planet
8  1842128       4         1  Action  Dinosaur Planet
9  2238063       3         1  Action  Dinosaur Planet
#Checking the shape of our data

DF.shape
(1048574, 5)
# Checking the information about our dataset

DF.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1048574 entries, 0 to 1048573
Data columns (total 5 columns):
 #   Column     Non-Null Count    Dtype 
---  ------     --------------    ----- 
 0   Cust_Id    1048574 non-null  int64 
 1   Rating     1048574 non-null  int64 
 2   Movie_Id   1048574 non-null  int64 
 3   Genre      1047488 non-null  object
 4   MovieName  1048574 non-null  object
dtypes: int64(3), object(2)
memory usage: 40.0+ MB
# Checking the missing values in our dataset

DF.isnull().sum()
Cust_Id         0
Rating          0
Movie_Id        0
Genre        1086
MovieName       0
dtype: int64
#Counting the Genere values in our dataset

DF["Genre"].value_counts()
Historical     240327
Animation      123898
Educational    111976
Mystery        100898
Crime           97323
Biography       88510
Gang            54861
War             43446
RomCom          39629
Documentary     38282
Horror          31271
Other           25436
Drama           19258
Fan             13278
Thriller         9588
Sci-Fi           7654
Fiction          1019
Action            547
Comedy            145
Romance           142
Name: Genre, dtype: int64
#Q1.Find out the list of most popular and liked genre
#Ans. Q1

DF['Genre'].value_counts().to_frame().idxmax()
Genre    Historical
dtype: object
DF.dropna(inplace=True)
DF.shape
(1047488, 5)
DF.isnull().sum()
Cust_Id      0
Rating       0
Movie_Id     0
Genre        0
MovieName    0
dtype: int64
DF.iloc[0].Genre
'Action'
new_df2 = DF[['Movie_Id','Genre','MovieName']]
new_df2
         Movie_Id   Genre           MovieName
0               1  Action     Dinosaur Planet
1               1  Action     Dinosaur Planet
2               1  Action     Dinosaur Planet
3               1  Action     Dinosaur Planet
4               1  Action     Dinosaur Planet
...           ...     ...                 ...
1048569       241  Horror  North by NorthWest
1048570       241  Horror  North by NorthWest
1048571       241  Horror  North by NorthWest
1048572       241  Horror  North by NorthWest
1048573       241  Horror  North by NorthWest

[1047488 rows x 3 columns]
new_df2['Genre'] = new_df2['Genre'].apply(lambda X:X.lower())
C:\Users\Acer\AppData\Local\Temp/ipykernel_3968/58045749.py:1: SettingWithCopyWarning: 
 .
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  new_df2['Genre'] = new_df2['Genre'].apply(lambda X:X.lower())
new_df2
         Movie_Id   Genre           MovieName
0               1  action     Dinosaur Planet
1               1  action     Dinosaur Planet
2               1  action     Dinosaur Planet
3               1  action     Dinosaur Planet
4               1  action     Dinosaur Planet
...           ...     ...                 ...
1048569       241  horror  North by NorthWest
1048570       241  horror  North by NorthWest
1048571       241  horror  North by NorthWest
1048572       241  horror  North by NorthWest
1048573       241  horror  North by NorthWest

[1047488 rows x 3 columns]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1047488, stop_words="english")
cv.fit_transform(new_df2["Genre"]).toarray().shape
(1047488, 20)
vectors=cv.fit_transform(new_df2["Genre"]).toarray()
vectors[0]
array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int64)
len(cv.get_feature_names())
20
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        return " ".join(y)
new_df2['Genre']=new_df2['Genre'].apply(stem)
C:\Users\Acer\AppData\Local\Temp/ipykernel_3968/886652446.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  new_df2['Genre']=new_df2['Genre'].apply(stem)
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)
---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
~\AppData\Local\Temp/ipykernel_3968/1403471309.py in <module>
----> 1 cosine_similarity(vectors)

~\anaconda3\lib\site-packages\sklearn\metrics\pairwise.py in cosine_similarity(X, Y, dense_output)
   1186         Y_normalized = normalize(Y, copy=True)
   1187 
-> 1188     K = safe_sparse_dot(X_normalized, Y_normalized.T,
   1189                         dense_output=dense_output)
   1190 

~\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
     61             extra_args = len(args) - len(all_args)
     62             if extra_args <= 0:
---> 63                 return f(*args, **kwargs)
     64 
     65             # extra_args > 0

~\anaconda3\lib\site-packages\sklearn\utils\extmath.py in safe_sparse_dot(a, b, dense_output)
    150             ret = np.dot(a, b)
    151     else:
--> 152         ret = a @ b
    153 
    154     if (sparse.issparse(a) and sparse.issparse(b)

MemoryError: Unable to allocate 7.98 TiB for an array with shape (1047488, 1047488) and data type float64
cosine_similarity(vectors).shape
similarity = cosine_similarity(vectors)
similarity[0]
similarity[0].shape
sorted(list(enumerate(similarity[0])), reverse = True, key=lambda X:X[1])[1:3]
def recommend(MovieName):
    Movie_Id = new_df2[new_df2['title']==MovieName].index[0]
    distances=similarity[MovieName_index]
    MovieName_list = sorted(list(enumerate(distances)), reverse=True, key=lambda X:X[1])[1:3]
    
    for i in MovieName_list:
        print(new_df2.iloc[i[0]].title)
    
recommend(Broken Blossoms)





--->
