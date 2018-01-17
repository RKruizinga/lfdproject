import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from nltk.tokenize import TweetTokenizer

from basicFunctions import basicFunctions

class svmClassifier:
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  Y_predicted = []
  labels = []

  def __init__(self, X_train, X_test, Y_train, Y_test, labels):
    self.X_train = X_train
    self.X_test = X_test
    self.Y_train = Y_train
    self.Y_test = Y_test

    self.labels = labels

  def classify(self):
    self.classifier = Pipeline([('feats', FeatureUnion([
	 					 ('char', TfidfVectorizer(tokenizer=Tokenizer.tweetIdentity, norm="l1", lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(tokenizer=Tokenizer.tweetIdentity, norm="l1", lowercase=False, analyzer='word', ngram_range=(1,3), min_df=1)),#, max_features=100000)),
      ])),
      ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=None))
    ])

    self.classifier.fit(self.X_train, self.Y_train)  

  def evaluate(self):
    self.Y_predicted = self.classifier.predict(self.X_test)
    self.accuracy, self.precision, self.recall, self.f1score = basicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    basicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    basicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)

class Tokenizer: #collection class of different tokenizers
  def tweetIdentity(arg):
	  tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	  return tokenizer.tokenize(arg)

