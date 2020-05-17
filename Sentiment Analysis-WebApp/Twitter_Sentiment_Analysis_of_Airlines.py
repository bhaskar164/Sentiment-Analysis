#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#Data Analysis............
twitter_train = pd.read_csv('twitter_train.csv')


# In[3]:


# twitter_test = pd.read_csv('twitter_test.csv')


# In[4]:


# twitter_train.shape


# In[5]:


# twitter_test.shape


# In[6]:


# twitter_train.head()


# In[7]:


# twitter_test.head()


# In[8]:


# twitter_train.isnull().sum()


# In[9]:


# twitter_train.columns


# In[10]:


# set(twitter_train['retweet_count'].values)


# In[11]:


# set(twitter_train['airline_sentiment'].values)


# In[12]:


# set(twitter_train['airline'].values)


# In[13]:


ex = []
ex.append([(twitter_train.loc[2,'text'], 'negative')])
ex.append([(twitter_train.loc[2,'text'], 'negative')])
ex[0]
len(ex)


# In[14]:


##Fetching Text and Sentiments for applying NLP on xtrain
def Fetch_Text_sent(data):
    tweet_text = []
    row = data.shape[0]
    for i in range(row):
        tweet_text.append((data.loc[i,'text'] , data.loc[i,'airline_sentiment']))
    return tweet_text


# In[15]:


##Fetching Text and Sentiments for applying NLP on xtest
def Fetch_Text_sent_xtest(data):
    tweet_text = []
    row = data.shape[0]
    for i in range(row):
        tweet_text.append(data.loc[i,'text'])
    return tweet_text


# In[16]:


##Checking function Fetch_Text_sent()
sample = twitter_train.iloc[0:5,0:]
xtrain = Fetch_Text_sent(sample)
len(xtrain)


# In[17]:


nlp_xtrain=[('','')]
nlp_xtrain = Fetch_Text_sent(twitter_train.iloc[0:,0:])
len(nlp_xtrain)


# In[18]:


nlp_xtest=[('','')]
nlp_xtest = Fetch_Text_sent_xtest(twitter_test.iloc[0:,0:])
len(nlp_xtest)


# In[19]:


from nltk import WordNetLemmatizer as WL
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize as WT
from nltk import pos_tag


# In[20]:


###Getting words from train and test data

xtrain_word = []
for text in nlp_xtrain:
    xtrain_word.append([WT(text[0]) , text[1]])


# In[21]:


xtest_word = []
for text in nlp_xtest:
    xtest_word.append(WT(text))


# In[22]:


xtest_word[0]


# In[23]:


import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
punc = list(string.punctuation)
stop = stop+punc


# In[24]:


def get_simple_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[25]:


#Function to remove stop words alphanums , applying lemmatizer
def clean_reviews(word):
    output_words = []
    for w in word:
        if ((w.lower() not in stop) and (w.isalpha())):
            pos = pos_tag([w])
            lemm = WL()
            lem_word = lemm.lemmatize(w,pos = get_simple_tag(pos[0][1]))
            output_words.append(lem_word.lower())
    return output_words


# In[26]:


#cleaning train data
clean_data = []
for doc in xtrain_word:
    text = clean_reviews(doc[0])
    if len(text)>0:
        clean_data.append((text,doc[1]))
clean_data[0:5]


# In[27]:


len(clean_data)


# In[28]:


#Cleaning Test Data
cleanTest_data = []
for doc in xtest_word:
    cleanTest_data.append(clean_reviews(doc))
cleanTest_data[0:5]


# In[29]:


len(cleanTest_data)


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
categories = [categ for doc,categ in clean_data] 


# In[31]:


categories[2]


# In[32]:


text_docs =  [" ".join(doc) for doc,categ in clean_data] 


# In[33]:


test_text_docs = [" ".join(doc) for doc in cleanTest_data] 


# In[34]:


test_text_docs[3]


# In[35]:


count_vec = CountVectorizer(stop_words = stop,max_features=3000)


# In[36]:


x_train = count_vec.fit_transform(text_docs)


# In[37]:


x_train.shape


# In[38]:


x_test = count_vec.transform(test_text_docs)
count_vec.get_feature_names()


# In[39]:


mat = x_train.todense()


# In[40]:


mat[5000,333]


# In[41]:


from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


# In[42]:


ytrain = np.reshape(categories, (10980,)).T


# In[43]:


x_train[0:1]


# In[57]:


# svmclf = svm.SVC()
# svmclf.fit(x_train,ytrain)
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
from sklearn.neural_network import MLPClassifier
rfc = MLPClassifier(hidden_layer_sizes=(200,20,10),max_iter=300)


# In[58]:


rfc.fit(x_train,ytrain)


# In[59]:


svmpred = rfc.predict(x_test)


# In[60]:


set(svmpred)


# In[61]:


import csv


with open('output.csv', 'w') as csvFile:
    writer = csv.writer(csvFile,delimiter=",")
    for r in svmpred:
      writer.writerow([r])

