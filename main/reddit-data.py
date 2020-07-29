import praw
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
import joblib
import re
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import random
from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s


reddit = praw.Reddit(client_id='#', client_secret='#', user_agent='#', username='#', password='#')
subreddit = reddit.subreddit('india')
flairs = ["AskIndia", "Non-Political", "[R]eddiquette", "Scheduled", "Photography", "Science/Technology", "Politics", "Business/Finance", "Policy/Economy", "Sports", "Food", "AMA"]
topics_dict = {"author":[],"body":[], "comment":[], "flair":[], "id":[], "score":[], "title":[], "url":[], "combined_features":[]}
k=0
'''data collection
for flair in flairs:
  
  get_subreddits = subreddit.search(flair, limit=200)
  
  for submission in get_subreddits:
    
    topics_dict["flair"].append(flair)
    topics_dict["author"].append(submission.author)
    topics_dict["body"].append(submission.selftext)
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_level_comment in submission.comments:
      comment = comment + ' ' + top_level_comment.body
    topics_dict["comment"].append(comment)
    combined_features = submission.title + ' ' + comment + ' ' + submission.url + ' ' + submission.selftext
    topics_dict["combined_features"].append(combined_features)
    k+=1
    print(k)
    
df = pd.DataFrame(topics_dict)
#df.dropna(self, 0)

#df.head(10)
print("df")

df['body'] = df['body'].apply(str)
df['title'] = df['title'].apply(str)
df['comment'] = df['comments'].apply(str)
df['url'] = df['url'].apply(str)
df['combined_features'] = df['combined_features'].apply(str)

df['body'] = df['body'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)
df['comment'] = df['comments'].apply(clean_text)
df['url'] = df['url'].apply(clean_text)
df['combined_features'] = df['combined_features'].apply(clean_text)

df.to_csv('data.csv', index=False) 
'''
data = pd.read_csv('data.csv')
data.fillna("",inplace = True)
'''#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
#stemmer.stem(abcd)
words = stopwords.words("english")

df['title'] = df['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['body'] = df['body'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['comments'] = df['comments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['url'] = df['url'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
#df['title'] = df['title'] + ' ' + df['body']
df['combined_features'] = df['combined_features'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
'''

#print("done")
###          Models    ##################
def nb_classifier(X_train, X_test, y_train, y_test):
  nb = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB()),
                ])
  nb.fit(X_train, y_train)

  y_pred = nb.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def linear_svm(X_train, X_test, y_train, y_test):

  from sklearn import svm
  sv = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', svm.SVC()),
                 ])
  sv.fit(X_train, y_train)

  y_pred = sv.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def logisticreg(X_train, X_test, y_train, y_test):

  from sklearn.linear_model import LogisticRegression

  logreg = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                 ])
  logreg.fit(X_train, y_train)

  y_pred = logreg.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def randomforest(X_train, X_test, y_train, y_test):
  ranfor = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42)),
                 ])
  ranfor.fit(X_train, y_train)

  y_pred = ranfor.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def train_test(X,y):
 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
  
  print("Results of Naive Bayes Classifier")
  nb_classifier(X_train, X_test, y_train, y_test)
  print("Results of Support Vector Machine")
  linear_svm(X_train, X_test, y_train, y_test)
  print("Results of Logistic Regression")
  logisticreg(X_train, X_test, y_train, y_test)
  print("Results of Random Forest")
  randomforest(X_train, X_test, y_train, y_test)


t = data.flair

a = data.title
c = data.comment
b = data.body
d = data.url
e = data.combined_features

print("Flair Detection using Title as Feature")
train_test(a,t)
print("Flair Detection using Body as Feature")
train_test(b,t)
print("Flair Detection using URL as Feature")
train_test(d,t)
print("Flair Detection using Comments as Feature")
train_test(c,t)
print("Flair Detection using combined_features as Feature")
train_test(e,t)

###  x = features ###
###  y = labels ###
X_train, X_test, y_train, y_test = train_test_split(e, t, test_size=0.3, random_state = 42)
model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42)),
                 ])
model.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)#('vect', CountVectorizer()),
