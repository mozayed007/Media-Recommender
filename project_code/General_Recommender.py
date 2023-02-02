#!/usr/bin/env python
# coding: utf-8

# # **1-Dataset Analysis:**
# 

# ## *i) Cleaning Dataset:* <br />
#    Jan 23 Last edits

# Latest check 30th of Jan. 2023 at 5:00PM

# ### **Importings:**

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from ast import literal_eval



# In[2]:


data= pd.read_csv("../data-history/up-to-date-MAL/anime_Feb23.csv")


print(data.shape)
data.head(1)


# In[3]:


data.columns


# ### Taking care of nulls and drops:

# In[4]:


drops=["main_picture_medium","main_picture_large","broadcast_day_of_the_week","broadcast_start_time","alternative_titles_en","alternative_titles_ja","alternative_titles_synonyms"]
data_main=data.drop(drops,axis=1)


# In[5]:


sum(data_main.isnull().sum())


# In[6]:


data_main.fillna(value=data['mean'].mean,inplace=True)
data_main.fillna(value=data['rank'].mean,inplace=True)
data_main.fillna(value=data['num_favorites'].mean,inplace=True)


# In[7]:


sum(data_main.isnull().sum())


# In[8]:



data_main.dropna(subset=['mean','source','num_episodes','start_date','end_date','rank','average_episode_duration','rating','start_season_year','synopsis'],inplace=True)
print(sum(data_main.isnull().sum()))
print(data_main.shape)


# In[10]:


data_main.nunique()


# In[11]:


data_main['status'].unique()


# ## Encoding and Adjusting Dtypes:
#  Using separate Data_Frame for reviewing, Yes, Enough ram is available.

# In[12]:


df=pd.get_dummies(data_main, columns=["media_type","status","source","nsfw","genres","rating","studios","start_season_season"], prefix=["media_type","status","source","nsfw","genres","rating","studios","start_season_season"])


# In[13]:


df.shape 


# ###  NLP Pre-processing

# #### **Applying Key-BERT for Keywords extraction:**

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
from tqdm.notebook import tqdm
import ast
import re
import spacy as sp
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


data_main.title.head(10)


# In[16]:


NLP = sp.load("en_core_web_lg")
TITLE = 'Death Note'
text = df[df['title'] == TITLE].synopsis.values[0]
key_model = KeyBERT()
df = df[~df.title.duplicated(keep='first')]
def clean_text(text):
    doc = re.sub("[\(\[].*?[\)\]]", "", text) # Remove the "written by" caption
    doc = doc.replace(u'\n', u'').replace(u'\r', u'')
    doc =  re.sub('[^a-zA-Z]', " ", text)
    doc = ' '.join(text.split())
    doc = text.lower()
    doc = NLP(doc)
    return doc

doc = clean_text(text)
print(doc)


# In[17]:


# Based on https://stackoverflow.com/questions/48925328/how-to-get-all-noun-phrases-in-spacy
def get_candidates(doc):
    # code to recursively combine nouns
    # 'We' is actually a pronoun but included in your question
    # hence the token.pos_ == "PRON" part in the last if statement
    # suggest you extract PRON separately like the noun-chunks above

    index = 0
    noun_indices = []
    for token in doc:
        if token.pos_ == 'NOUN':
            noun_indices.append(index)
        index = index + 1

    #print('Nouns found: ', len(noun_indices))

    candidates = []
    for idxValue in noun_indices:
        if not bool(doc[idxValue].left_edge.ent_type_):
            start = doc[idxValue].left_edge.i
        else:
            start = idxValue 

        if not bool(doc[idxValue].right_edge.ent_type_):
            finish = doc[idxValue].right_edge.i+1
        else:
            finish = idxValue + 1

        if finish-start > 0 and finish-start <7:
            span = doc[start : finish]
#             print('>', span)
            candidates.append(span.text)

    return candidates

candidates = get_candidates(doc)
print(candidates)


# In[ ]:


keywords = key_model.extract_keywords(doc.text, candidates=candidates, 
                                use_mmr=True, diversity=0.5)

print(keywords)


# * Creating clean text, nouns and keywords from synopsis.
# * Separate in new df for data analysis.
# * Delete Syns entries from main df.
# 

# In[ ]:


df.loc[:, 'cleaned_syn'] = df.loc[:,'synopsis'].apply(clean_text)


# In[ ]:


df.loc[:, 'nouns'] = df.loc[:,'cleaned_syn'].apply(get_candidates)


# In[ ]:


syns=['id','title','synopsis','cleaned_syn','nouns']
dfsyn=df[syns]
dfsyn['genres']=data_main['genres']
dfsyn.head(10)


# In[ ]:


dropssyn=['synopsis','cleaned_syn','nouns']
df=df.drop(dropssyn,axis=1)


# In[ ]:


dfsyn['cleaned_syn'].values[3]


# In[ ]:


dfsyn.loc[:, 'doc_clean'] = dfsyn.loc[:,'cleaned_syn'].apply(NLP)


# In[ ]:


dfsyn['doc_clean'].values[3]


# In[ ]:


#dfsyn.drop('cleaned_syn',inplace=True,axis=1)
dfsyn.columns


# In[ ]:


#something weird
keyword= [None] * len(dfsyn)
docs= [None] * len(dfsyn)
for i  in range(len(dfsyn)):
    docs[i]= dfsyn["doc_clean"].values[i].text
    nouns = dfsyn["nouns"]
docs[3]


# In[ ]:



for i  in range(len(dfsyn)) :   
    keyword[i] = key_model.extract_keywords(docs=docs[i],candidates=nouns.values[i],use_mmr=True, diversity=0.6)
keywordunp= [None]* len(keyword)


# In[ ]:


import itertools
for i in range(len(keyword)):

        keywordunp[i]=list(itertools.chain(*keyword[i]))
        
len(keyword),len(keywordunp)


# In[ ]:


keywordunp[10]
keywords= [1]* len(keyword)


# In[ ]:



for i in range(len(keywordunp)):
    for j in range(len(keyword[i])):
        keywords[i]=keywordunp[i][0::2]


# In[ ]:


len(keywords),keywords[3]


# In[ ]:


extracted_keywords=np.array(keywords)


# In[ ]:


dfsyn['keywords']=extracted_keywords.tolist()
dfsyn['genres']=data_main['genres']
dfsyn['rating']=data_main['rating']
dfsyn['media_type']=data_main['media_type']
dfsyn.head(4)


# In[ ]:


dfsyn.columns


# In[ ]:


dfsyn.tail(1)


# In[ ]:


#dfsynsave=dfsyn.to_csv("../data-history/up-to-date-MAL/anime-synopsis-keywords-nlp.csv",index=False)


# ## *ii)EDA:*

# In[ ]:


data_main.columns


# In[ ]:


sns.set_style("dark")
plt.figure(figsize=(12,8))
plt.hist(df['mean'], bins=70,)
plt.show


# In[ ]:


fig = px.pie(data_main, 'media_type')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# *Notes:* Naturally TV has higher percentage as anime media.
# 

# In[ ]:


corr = data_main.corr()

# Set up the matplotlib plot configuration
#
f, ax = plt.subplots(figsize=(16, 10))
#
# Generate a mask for upper traingle
#
mask = np.triu(np.ones_like(corr, dtype=bool))
#
# Configure a custom diverging colormap
#
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# Draw the heatmap
#
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)


# *Notes:*  So, basically interesting factors that are affecting the mean factor are : rank, popularity, num_scoring_users, ignore num_list_users for now till further investigation of difference.

# In[ ]:


fig = px.histogram(data_main[data_main['start_date'].dt.year >= 1980], x='start_date', color='media_type')
fig.update_layout(bargap=0.1)


# *Notes:* Obviously 2016 was a good year for Otakus :3 specially summer-Autumn-Fall seasons, with 119 tv, 45 movie, 23 ova, 61 ona, 60 special and 41 music. (Gotta check watching list lmao).

# In[ ]:


data_main.groupby('num_episodes')['id'].count().sort_values(ascending=False).head(30).plot(kind='bar', figsize=(15,10))
plt.show()


# *Notes:* A lot of Movies (1 episode) that's why the spike, but the summation of all others are the other percentages of tv,ova,ona,... etc. most tv/specials are short 12 (episodes)/(season|title). </br>
# *The fans of "When you have eliminated the impossible" teenager for 22+ years don't give up :(* </br>
# *Gomu Gomu no guys don't be Sadge :(*

# # 2- **MODELS TIME:** :3
# ![image info](https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/gettyimages-458406992-1538405221.jpg?crop=0.9xw:0.9xh;0,0&resize=256:*) <br />
#   July's 2022 Work

# ## Similarity Analysis :

# ### Synopsis Keyword Analysis:
# *(NLP)* :
# * KeyBERT.
# * Spacy.
# * tqdm.
# * CountVectorizer.
# * TF-IDF

# In[ ]:


import sweetviz as sv
#You could specify which variable in your dataset is the target for your model creation. We can specify it using the target_feat parameter.
data_report = sv.analyze(data)


# In[ ]:


data_report.show_notebook(w=1500, h=900, scale=0.8)
data_report.show_html(scale=0.9)


# ### Cos-similarity:

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


# In[ ]:


tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['synopsis'] + data['genres'] + data['rating'] + data['studios']+data['media_type'])
tfidf_matrix.shape


# Using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. 
# 
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||}$

# In[ ]:


cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


data = data_main.reset_index()
titles = data['title']
indices = pd.Series(data_main.index, index=data['title'])


# In[ ]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    anime_indices = [i[0] for i in sim_scores]
    return titles.iloc[anime_indices]
data['title'][3]


# In[ ]:


cos_results=get_recommendations('Death Note').head(10)
cos_results


# Not so close recommendations but good start
# 

# #### **Featuring keywords and similarities:**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(","), binary='true')
#vectors=[None]*len(keywords)


# In[ ]:


dfsyn['keywords'].head(4)


# In[ ]:


dfsyn.head()


# In[ ]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['genres', 'keywords', 'rating', 'media_type']

for feature in features:
    dfsyn[feature] = dfsyn[feature].apply(clean_data)


# In[ ]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['genres']) + ' ' + x['rating'] + ' ' + ' '.join(x['media_type'])
dfsyn['soup'] = dfsyn.apply(create_soup, axis=1)


# In[ ]:


dfsyn['soup'].head()


# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cos_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cos_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return dfsyn['title'].iloc[anime_indices]


# In[ ]:




count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dfsyn['soup'])



cos_sim2 = cosine_similarity(count_matrix, count_matrix)
dfsyn = dfsyn.reset_index()
indices = pd.Series(dfsyn.index, index=dfsyn['title'])


# In[ ]:


cos2_results=get_recommendations('Death Note', cos_sim2)
len(cos2_results),cos2_results


# **Cos-1 is genertated from similarity of multiple features using TF-IDF one of them was the whole synopsis of animes, while Cos2 was using keywords of synopsis instead of the whole synopsis feature**

# In[ ]:


from keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten

import keras
from keras import layers 
from keras.models import Model
from keras.optimizers import Adam


# In[ ]:


rdf=pd.read_csv(r"../data-history/ratings-2020/rating_complete.csv")
n = 10

# Count the lines or use an upper bound
num_lines = sum(1 for l in open(r"../data-history/ratings-2020/rating_complete.csv"))

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = [x for x in range(1, num_lines) if x % n != 0]

# Read the data
#rdf = pd.read_csv(r"../data-history/ratings-2020/rating_complete.csv", skiprows=skip_idx )
print(rdf.shape)
rdf.columns


# ### **Pair wise distance** :

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances


# In[ ]:


# Removing Duplicated Rows
duplicates = rdf.duplicated()

if duplicates.sum() > 0:
    print('> {} duplicates'.format(duplicates.sum()))
    rdf = rdf[~duplicates]

print('> {} duplicates'.format(rdf.duplicated().sum()))


# In[ ]:


rdf.columns


# In[ ]:


# Scaling BTW (0 , 1.0)
min_rating = min(rdf['rating'])
max_rating = max(rdf['rating'])
rdf['rating'] = rdf["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values.astype(np.float64)

AvgRating = np.mean(rdf['rating'])
print('Avg', AvgRating)


# In[ ]:


rdf.head(2)


# In[ ]:


g = rdf.groupby('user_id')['rating'].count()
top_users = g.dropna().sort_values(ascending=False)[:20]
top_r = rdf.join(top_users, rsuffix='_r', how='inner', on='user_id')

g = rdf.groupby('id')['rating'].count()
top_animes = g.dropna().sort_values(ascending=False)[:20]
top_r = top_r.join(top_animes, rsuffix='_r', how='inner', on='id')

pd.crosstab(top_r.user_id, top_r.id, top_r.rating, aggfunc=np.sum)


# In[ ]:


top_r.isnull().sum()


# In[ ]:


anime_ids = rdf["id"].unique().tolist()
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
n_animes = len(anime2anime_encoded)
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}

rdf["anime"] = rdf["id"].map(anime2anime_encoded)

user_ids = rdf["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
n_users = len(user2user_encoded)
rdf["user"] = rdf["user_id"].map(user2user_encoded)


print("Num of users: {}, Num of animes: {}".format(n_users, n_animes))
print("Min rating: {}, Max rating: {}".format(min(rdf['rating']), max(rdf['rating'])))


# In[ ]:


tmatrix = np.zeros((n_users, n_animes))
tmatrix.shape


# In[ ]:


rdf.columns, rdf.shape


# In[ ]:


for line in rdf.itertuples():
    tmatrix[line[5]-1, line[4]-1] = line[3]


# In[ ]:


tmatrix[1]


# In[ ]:


user_distances = pairwise_distances(tmatrix, metric="cosine")

# ".T" below is to transpose our 2D matrix.
tmatrix_transpose = tmatrix.T
anime_distances = pairwise_distances(tmatrix_transpose, metric="cosine")

user_distances.shape, anime_distances.shape


# In[ ]:


user_similarity = 1 - user_distances
anime_similarity = 1 - anime_distances


# In[ ]:


data_main.columns


# In[ ]:


idx_to_anime = {}
for line in data_main.itertuples():
        idx_to_anime[(line[1])-1] = line[3]
anime_to_idx = {v: k for k, v in idx_to_anime.items()}


# In[ ]:


anime_idx= anime_to_idx['Death Note']

def top_k_similar(similarity, mapper , anime_idx, k=8):
      return [mapper[x] for x in np.argsort(similarity[anime_idx,:])[:-k-2:-1]]


# In[ ]:


pair_results= top_k_similar(anime_similarity,idx_to_anime ,anime_idx,k=8)
print(pair_results)


# In[ ]:


def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j
print('pair vs cos2')
print(jaccard_similarity(pair_results,cos2_results))
print('cos1 vs cos2')
print(jaccard_similarity(cos_results,cos2_results))


# **zenzen wakaranaaaaaiiiii !!!!!!!!!!!!!** </br>
# :"D </br>
# pair-wise distance results not related to cosine Similarity results at all no intersections. </br>
# using keywords or using Full synopsis didn't matter for cos similarity so better for resources use keywords

# ### **RecommendNet Maybe?** :

# **Zenzen heiki janai :"D , Tasukete, Dare ka tasukeeteeeeee !** <br />
#   Aug 2022 Work

# In[ ]:


rdf.shape,rdf.isnull().sum(),len(rdf['user'])


# In[ ]:


df.filter(regex='^media_type_',axis=1).head(2), df.filter(regex='^source_',axis=1).head(2)


# In[ ]:


data['id'].values[1],data['popularity'].values[1]


# #### *Normal Recommender features*

# **Collaborative Filtering Approach, dependancy on user rates**

# In[ ]:


# Shuffle
rdf = rdf.sample(frac=1, random_state=73)

X = rdf[['user', 'anime']].values
y = rdf["rating"]


# In[ ]:


# Split
test_set_size = 10000 #10k for test set
train_indices = rdf.shape[0] - test_set_size 

X_train, X_test, y_train, y_test = (
    X[:train_indices],
    X[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

print('> Train set ratings: {}'.format(len(y_train)))
print('> Test set ratings: {}'.format(len(y_test)))


# In[ ]:


X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


# In[ ]:


print(len(X_train))
print(len(y_train))


# In[ ]:


X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


# In[ ]:


print(len(X_train_array[0]))
print(len(X_test_array[0]))


# In[ ]:


def recommender_net():
    embedding_size = 64
    
    user = Input(name = 'user', shape = [1])
    user_embedding = Embedding(name = 'user_embedding',
                    input_dim = n_users, 
                    output_dim = embedding_size)(user)
    
    anime = Input(name = 'anime', shape = [1])
    anime_embedding = Embedding(name = 'anime_embedding',
                    input_dim = n_animes, 
                    output_dim = embedding_size)(anime)
    #x = Concatenate()([user_embedding, anime_embedding])
    x = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, anime_embedding])
    x = Flatten()(x)
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='adam')
    
    return model

model1 = recommender_net()
model1.summary()


# In[ ]:





# In[ ]:


# Callbacks
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
batch_size = 10000
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

checkpoint_filepath = './weights.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)

early_stopping = EarlyStopping(patience = 3, monitor='val_loss', 
                            mode='min', restore_best_weights=True)

my_callbacks = [
    model_checkpoints,
    lr_callback,
    early_stopping,   
]


# In[ ]:


print(len(X_test_array[0]))
print(len(y_test))


# In[ ]:


# Model training
history = model1.fit(
    x=X_train_array,
    y=y_train,
    batch_size=batch_size,
    epochs=20,
    verbose=1,
    validation_data=(X_test_array, y_test),
    callbacks=my_callbacks
)

model1.load_weights(checkpoint_filepath)


# In[ ]:


#Training results
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history["loss"][0:-2])
plt.plot(history.history["val_loss"][0:-2])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


# In[ ]:


from tqdm.keras import TqdmCallback


history = model1.fit(
    x=X_train_array,
    y=y_train,
    batch_size=batch_size,
    epochs=30,
    validation_data=(X_test_array, y_test),
    verbose = 0, 
    callbacks=[TqdmCallback(verbose=0)])

model1.load_weights(checkpoint_filepath)


# In[ ]:


#Training results
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history["loss"][0:-2])
plt.plot(history.history["val_loss"][0:-2])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


# In[ ]:


def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

anime_weights = extract_weights('anime_embedding', model1)
user_weights = extract_weights('user_embedding', model1)


# In[ ]:


data_main.columns


# In[ ]:


name = data[data_main.id == 100].title.values[0]
print(name)


# In[ ]:


# Fixing Names
def get_animename(anime_id):
    try:
        name = data[data_main.id == anime_id].title.values[0]
        return name
    except:
        print('error')
        return 0

data["eng_version"] = data['title']


data_main.sort_values(by=['mean'], 
                inplace=True,
                ascending=False, 
                kind='quicksort',
                na_position='last')

df = data[["id","title", "mean", "genres", "num_episodes", 
        "media_type","synopsis"]]


def get_animeframe(anime):
    if isinstance(anime, int):
        return df[df.id == anime]
    if isinstance(anime, str):
        return df[df.title == anime]
def get_sypnopsis(anime):
    if isinstance(anime, int):
        return df[df.id == anime].synopsis.values[0]


# In[ ]:


df.shape


# In[ ]:


pd.set_option("max_colwidth", None)

def find_similar_animes(name, n, return_dist=False, neg=False):
        index = get_animeframe(name).id.values[0]
        print(index)
        encoded_index = anime2anime_encoded.get(index)
        weights = anime_weights
        print(encoded_index)
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1            
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        print('animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        rindex = df
        similarityarr = []
        for close in closest:
            decoded_id = anime_encoded2anime.get(close)
            sypnopsis = get_sypnopsis(decoded_id)
            anime_frame = get_animeframe(decoded_id)
            anime_name = anime_frame.title.values[0]
            genre = anime_frame.genres.values[0]
            similarity = dists[close]
            similarityarr.append({"id": decoded_id, "title": anime_name,
                            "similarity": similarity,"genres": genre,
                            'synopsis': sypnopsis})
        frame = pd.Dataframe(similarityarr).sort_values(by="similarity", ascending=False)
        return frame[frame.id != index].drop(['id'], axis=1)


# In[ ]:


find_similar_animes('Death Note', n=10, neg=False)


# #### *Features modding* <br />
#    Modifying parameters for Recommend NET

# In[ ]:


# dfdl =pd.DataFrame()


# In[ ]:


# dfdl_ids = data["id"].tolist()
# dfdlid_encoded = {x: i for i, x in enumerate(dfdl_ids)}
# n_animes = len(dfdlid_encoded)
# id_encoded2id = {i: x for i, x in enumerate(dfdl_ids)}
# dfdl["id"] = data["id"].map(dfdlid_encoded)

# dfdl_mean = data["mean"].tolist()
# dfdl_mean_encoded = {x: i for i, x in enumerate(dfdl_mean)}
# mean_encoded2mean = {i: x for i, x in enumerate(dfdl_mean)}
# n_users = len(dfdl_mean_encoded)
# dfdl["mean"] = data["mean"].map(dfdl_mean_encoded)

# dfdl_pop = data["popularity"].tolist()
# user2user_encoded = {x: i for i, x in enumerate(dfdl_pop)}
# user_encoded2user = {i: x for i, x in enumerate(dfdl_pop)}
# n_users = len(user2user_encoded)
# dfdl["popularity"] = data["popularity"].map(user2user_encoded)


# In[ ]:


data_main.columns, data_main.shape


# In[ ]:



# x1 = rdf[['user', 'anime']].values 

# #x2=  data[['id'],['popularity']].values
# x3=data[['mean'],['num_scoring_users']].values
# x4=data['rank'].tolist(),data['num_favorites'].tolist()
# x5= df.filter(regex='^media_type_',axis=1).values[i]
# x6= df.filter(regex='^source_',axis=1).values[i]

# y = rdf["rating"]
# # Split
# test_set_size = 250000 #10k for test set
# train_indices = rdf_sampled.shape[0] - test_set_size 
# len(x1),len(x2),len(x2[1]),len(x3),len(x3[1]),len(x4),len(x4[1]),len(y),


# In[ ]:


# X3= x3[:,0] + x3[:,2] +x3[:,3] + x3[:,4] + x3[:,5] + x3[:,1] 
# X4=['None']*len(x4)*len(x4[1])
# for i in range(len(x4[1])):
#     X4 =X4 + x4[:,i]


# In[ ]:


# X1_train, X1_test, y_train, y_test = (
#     x1[:train_indices],
#     x1[train_indices:],
#     y[:train_indices],
#     y[train_indices:],
# )


# ### *After Reading Some Articels:*

# #### **Research at home** <br />
#    Dec. 2022 work <br />
# 
# Semantic Similarity on synopsis using nlp models.

# Potential Models for learning: <br />
# * paraphrase-miniLM
# * stsb-roberta latest alternatives
# * bert-base-nli-mean-tokens

# **To_Do:**
# - Get embeddings from pretrained for all synopsis ( all paragraphs ).
# - Compare Similarity using distance wise / cosine / pairwise whatever the hell will measure similarity of embeddings.
# - Worst case senario ,(For each sentence embeddings in the requested anime synopsis loop cosine similarity between all sentences in all other synopsis)
# - Optimization worth testing: Finding similarity between sentences in the same synopsis to get unique sentences and store those while ignoring sentences that are pretty much similar in embeddings, that leads to having smaller group of sentences for each synopsis to loop on (Still looping bratan).
# - 5Head IDEA: Semantic Keyword embeddings similarity analysis to get potential chosen titles to do semantic sentence analysis on.
# - **OR JUST USE PARAPHRASE MINING U Fokin IDIOT, anata BAKA ??? hontoni BAKAAAA.**

# In[ ]:





# In[ ]:




