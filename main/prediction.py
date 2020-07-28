import sklearn
import pickle
import praw
import joblib
import re
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

reddit = praw.Reddit(client_id='anw_EpVABeFTzg', client_secret='CuEWY3Fmt1CIE1lrI12ypbfjGrc', user_agent='flair_detector', username='dev_sid', password='hitechcamp')
loaded_model = joblib.load('main/finalized_model.sav')

def detect_flair(url,loaded_model):

  submission = reddit.submission(url=url)

  data = {}

  submission.comments.replace_more(limit=None)
  comment = ''
  for top_level_comment in submission.comments:
    comment = comment + ' ' + top_level_comment.body
  data['title'] = str(submission.title)
  data['body'] = str(submission.selftext)
  data['url'] = str(submission.url)
  data['comment'] = str(comment)
  data['combine'] = clean_text(data['title']) + clean_text(data['comment']) + clean_text(data['url']) + clean_text(data['body'])
  
  return loaded_model.predict([data['combine']])
