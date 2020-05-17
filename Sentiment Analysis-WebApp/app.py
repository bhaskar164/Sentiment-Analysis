from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer as WL
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize as WT
from nltk import pos_tag
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from collections import Counter 


stop = stopwords.words('english')
punc = list(string.punctuation)
stop = stop+punc

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

def clean_reviews(word):
        output_words = []
        for w in word:
            if ((w.lower() not in stop) and (w.isalpha())):
                pos = pos_tag([w])
                lemm = WL()
                lem_word = lemm.lemmatize(w,pos = get_simple_tag(pos[0][1]))
                output_words.append(lem_word.lower())
        return output_words

count_vec = CountVectorizer(stop_words = stop,max_features=3000)
rfc = MLPClassifier(hidden_layer_sizes=(200,20,10),max_iter=300)


def training():
    print("Inside Training...............")
    twitter_train = pd.read_csv('twitter_train.csv')

    def Fetch_Text_sent(data):
        tweet_text = []
        row = data.shape[0]
        for i in range(row):
            tweet_text.append((data.loc[i,'text'] , data.loc[i,'airline_sentiment']))
        return tweet_text

    nlp_xtrain=[('','')]
    nlp_xtrain = Fetch_Text_sent(twitter_train.iloc[0:,0:])

    print("Fetching words from training data..................")
    xtrain_word = []
    for text in nlp_xtrain:
        xtrain_word.append([WT(text[0]) , text[1]])

    print("Fetching stop words.............................")
    # stop = stopwords.words('english')
    # punc = list(string.punctuation)
    # stop = stop+punc


    # def get_simple_tag(tag):
    #     if tag.startswith('J'):
    #         return wordnet.ADJ
    #     elif tag.startswith('V'):
    #         return wordnet.VERB
    #     elif tag.startswith('N'):
    #         return wordnet.NOUN
    #     elif tag.startswith('R'):
    #         return wordnet.ADV
    #     else:
    #         return wordnet.NOUN


    # def clean_reviews(word):
    #     output_words = []
    #     for w in word:
    #         if ((w.lower() not in stop) and (w.isalpha())):
    #             pos = pos_tag([w])
    #             lemm = WL()
    #             lem_word = lemm.lemmatize(w,pos = get_simple_tag(pos[0][1]))
    #             output_words.append(lem_word.lower())
    #     return output_words

    print("Cleaning training data.............................")
    #cleaning train data
    clean_data = []
    for doc in xtrain_word:
        text = clean_reviews(doc[0])
        if len(text)>0:
            clean_data.append((text,doc[1]))

    # print("Cleaning Test Data..................................")
    # #Cleaning Test Data
    # cleanTest_data = []
    # cleanTest_data.append(clean_reviews(test_data))


    categories = [categ for doc,categ in clean_data] 

    text_docs =  [" ".join(doc) for doc,categ in clean_data] 

    # test_text_docs = [" ".join(doc) for doc in cleanTest_data] 

    # print("Initialize count vectorizer................................")
    # count_vec = CountVectorizer(stop_words = stop,max_features=3000)

    x_train = count_vec.fit_transform(text_docs)

    # x_test = count_vec.transform(test_text_docs)

    ytrain = np.reshape(categories, (10980,)).T

    # print("Declaring Object of Classifier...................")
    # rfc = MLPClassifier(hidden_layer_sizes=(200,20,10),max_iter=300)
    print("Training the data.....................................")
    rfc.fit(x_train,ytrain)
    # print("Predicting............................................")
    # svmpred = rfc.predict(x_test)
    # print("Returning Predictions......................")
    # return svmpred[0]


# print("Training begins.........")
# training()

def testing_module(test_data):
    #Cleaning Test Data
    xtest_word = []
    xtest_word = WT(test_data)
    print("DATA...... ",xtest_word)
    cleanTest_data = []
    cleanTest_data = clean_reviews(xtest_word)
    print("CLEANED Test_data",cleanTest_data)
    test_text_docs = ["".join(doc) for doc in cleanTest_data] 
    print("AFTER CLEANING SENTENCE IS..........",test_text_docs)
    x_test = count_vec.transform(test_text_docs)
    print("SHAPE OF XTEST",x_test.shape)
    svmpred = rfc.predict(x_test)
    print("LENGTH OF PRED.... ",len(svmpred))
    pred_count = Counter(svmpred)
    print("COUNTS OF PRED... ",pred_count)
    pos=0
    neg=0
    neu=0
    for i in range(len(svmpred)):
        if(svmpred[i]=='positive'):
            pos+=1
        elif svmpred[i] =='negative':
            neg+=1
        else:
            neu+=1
    if(pos>neg and (pos>neu or pos>0)):
        ans = 'positive'
    elif(neg>pos and (neg>neu or neg>0)):
        ans = 'negative'
    else:
        ans = 'neutral'
    return ans



####################################################################################
####################################################################################
####################################################################################
####################################################################################

app = Flask( __name__ )

@app.route('/')
def index():
    print("Training begins.........")
    training()
    return render_template('index.html')
    
@app.route('/submit',methods=['POST'])
def submit():
    if request.method == 'POST':
        test_data  = request.form['data']
        output = testing_module(test_data)
    print("Output is................  ",output)
    if(output=='positive'):
        return render_template('output.html',message="positive")
    elif(output=='negative'):
        return render_template('output.html',message="negative")
    else:
        return render_template('output.html',message="neutral")

if __name__ == '__main__':
    app.debug = True
    app.run()