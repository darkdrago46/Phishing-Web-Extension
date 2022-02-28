#Machine Learning Model for the prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.pipeline import make_pipeline

from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

print('imported')

import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('phishing_site_urls.csv')

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

import time
print("Getting words tokenized...")
t0=time.time()
df['text_tokenized']=df.URL.map(lambda t: tokenizer.tokenize(t))
t1=time.time() -t0
print(f"Time taken {round(t1,2)} seconds")

stemmer = SnowballStemmer("english")
print("Getting words stemmed....")
t2=time.time()
df['text_stemmed'] = df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t3 = time.time()-t2
print(f"Time taken {round(t3,2)} seconds")
print("Getting joining words")
t4=time.time()
df['text_from_url'] = df['text_stemmed'].map(lambda i:' '.join(i))
t5=time.time()-t4
print(f"Time taken {round(t5,2)} seconds")


good_sites = df[df.Label=="good"]
bad_sites = df[df.Label=="bad"]

cv = CountVectorizer()
feature = cv.fit_transform(df.text_from_url)
X_train,X_test,y_train,y_test = train_test_split(feature,df.Label)
lr = LogisticRegression()
lr.fit(X_train,y_train)

pipeline_ls = make_pipeline(
    CountVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'),
    LogisticRegression()
)
X_train,X_test,y_train,y_test = train_test_split(df.URL,df.Label,test_size=0.2)
pipeline_ls.fit(X_train,y_train)

import pickle #to export model
pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))

result = loaded_model.score(X_test,y_test)
print(result)

#To test the model:

#predict=input("ENTER URL:")
#predict=[predict]
#loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#result = loaded_model.predict(predict)
#print(result)