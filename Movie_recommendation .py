#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np


# In[103]:


# read csv metadata file
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[104]:


movies.head(3)


# In[105]:


credits.head(3)


# In[106]:


# combine two files
movies = movies.merge(credits,on='title')


# In[107]:


movies.head(3)


# In[108]:


movies.info()


# In[109]:


# took necessary columns for analysis
movies = movies[['movie_id','title','genres','keywords','overview','cast','crew']]


# In[110]:


movies.head()


# In[111]:


movies.isnull().sum()


# In[112]:


# Remove null values
movies.dropna(inplace=True)


# In[113]:


# check duplicate values
movies.duplicated().sum()


# In[114]:


movies.iloc[0].genres


# In[115]:


import ast
def convert(obj):
    genr=[]
    for i in ast.literal_eval(obj):
        genr.append(i['name'])
    return genr    


# In[116]:


movies['genres'] = movies['genres'].apply(convert)


# In[117]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[118]:


def convert2(obj):
    actr=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=5:
            actr.append(i['name'])
            counter+=1
        else:
            break       
    return actr  


# In[119]:


movies['cast'] = movies['cast'].apply(convert2)


# In[120]:


def convert3(obj):
    dirctr=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            dirctr.append(i['name'])
            break
    return dirctr                   


# In[121]:


movies['crew'] = movies['crew'].apply(convert3)


# In[122]:


movies.head()


# In[123]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[124]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[125]:


movies.head()


# In[126]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[127]:


movies=movies[['movie_id','title','tags']]
movies['tags']=movies['tags'].apply(lambda x:" ".join(x))
movies.head()


# In[128]:


movies['tags']=movies['tags'].apply(lambda x:x.lower())


# In[129]:


movies.head()


# In[130]:


import nltk
from nltk.stem import  PorterStemmer,SnowballStemmer
port=PorterStemmer() 
#lema=WordNetLemmatizer()


# In[131]:


movies['tags'] = movies['tags'].apply(lambda x: " ".join(port.stem(i) for i in x.split()))


# In[132]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(strip_accents='ascii',max_features=7000,stop_words='english')


# In[133]:


vector=cv.fit_transform(movies['tags']).toarray()


# In[134]:


cv.get_feature_names()


# In[135]:


vector


# In[136]:


from sklearn.metrics.pairwise import cosine_similarity


# In[137]:


similarity = cosine_similarity(vector)


# In[172]:


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:7]
    
    for i in movies_list:
        #print(i)
        print(movies.iloc[i[0]].title)


# In[173]:


recommend('Iron Man')


# In[174]:


recommend('Batman Begins')


# In[157]:


similarity[1]


# In[141]:


import pickle


# In[88]:


pickle.dump(movies,open('movies_data.pkl','wb'))


# In[89]:


pickle.dump(movies.to_dict,open('movies_dict.pkl','wb'))


# In[158]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[152]:


movies_list = pickle.load(open('movies_data.pkl','rb'))
#movies1 = pd.DataFrame(movies_list())


# In[154]:


movies_list


# In[151]:


movies1['title'].values

