import os

import xml.etree.ElementTree as ET

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold


from nltk.tokenize import TweetTokenizer

class data:
  documents = {} #documents per language and per user
  labels = {}

  X = []
  Y = { 'age': [],
        'gender': [],
        'language': [] }

  def __init__(self, folders):
    self.readFiles(folders)
    self.languages = folders

  def readFiles(self, folders):
    for folder in folders:
      folder_path = 'training/'+folder
      files = os.listdir(folder_path)

      if folder not in self.documents:
        self.documents[folder] = {}
        
      if folder not in self.labels:
        self.labels[folder] = {}

      for file in files:
        if file.endswith(".xml"):
            tree = ET.parse(folder_path + '/' +file)
            root = tree.getroot()
            for child in root: #get all utterances of a person
              if child.tag == 'document':
                child.text = child.text.strip('\t') #strip all tabs
                if file[:-4] in self.documents[folder]:
                  self.documents[folder][file[:-4]].append(child.text)
                else:
                  self.documents[folder][file[:-4]] = [child.text]
        if file.endswith(".txt"):
          with open(folder_path + '/' +file) as f:
            lines = f.read()
            lines = lines.split('\n')
            for line in lines:
              if line != '':
                line = line.split(':::')
                self.labels[folder][line[0]] = (line[1], line[2]) # tuple, format: (gender, age)

  ### To collect all tweets from all specified languages, insert 'all'
  ### To collect tweets for a specific language, enter the language
  def collectXY(self, languages = 'all'):
    if languages == 'all':
      for language in self.languages:
        for user in self.documents[language]:
          for document in self.documents[language][user]:
            self.X.append(document)
            self.Y['gender'].append(self.labels[language][user][0])
            self.Y['age'].append(self.labels[language][user][1])
            self.Y['language'].append(language)
    elif languages in self.languages:
      for user in self.documents[languages]:
        for document in self.documents[languages][user]:
          self.X.append(document)
          self.Y['gender'].append(self.labels[languages][user][0])
          self.Y['age'].append(self.labels[languages][user][1])
          self.Y['language'].append(languages)

class svmClassifier:
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  predictedY = []
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
    self.predictedY = self.classifier.predict(self.X_test)

    self.accuracy = np.mean(self.predictedY == self.Y_test)
    self.precision = sklearn.metrics.precision_score(self.Y_test, self.predictedY, average="macro")
    self.recall = sklearn.metrics.recall_score(self.Y_test, self.predictedY, average="macro")
    self.f1score = sklearn.metrics.f1_score(self.Y_test, self.predictedY, average="macro")

  def printBasicEvaluation(self):    
    basicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    print("Class \t Precision \t Recall \t F-score")

    for label in self.labels:
      precisionScore = sklearn.metrics.precision_score(self.Y_test, self.predictedY, average="macro", labels=label)
      recallScore = sklearn.metrics.recall_score(self.Y_test, self.predictedY, average="macro", labels=label)
      f1Score = sklearn.metrics.f1_score(self.Y_test, self.predictedY, average="macro", labels=label)

      print(label, "\t", round(precisionScore,3), "\t\t", round(recallScore,3), "\t\t", round(f1Score,3), "\t")

class Tokenizer: #collection class of different tokenizers
  def tweetIdentity(arg):
	  tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	  return tokenizer.tokenize(arg)

class basicFunctions:
  def avg(l):
    return sum(l) / len(l)

  def printEvaluation(accuracy, precision, recall, f1score, text):    
    print("~~~" + text + "~~~ \n")
    print("Accuracy:\t {}".format(round(accuracy, 3)))
    print("Precision:\t {}".format(round(precision, 3)))
    print("Recall:\t\t {}".format(round(recall, 3)))
    print("F1-Score:\t {}".format(round(f1score, 3)))
  
### OPTIONS BEGIN
languages = ['dutch', 'english', 'spanish', 'italian']

collect_languages = 'all' #which language of documents
label_prediction = 'age' #predict age or gender

use_cross_validation = False #option to toggle between cross validation and simple validation
K_folds = 5
### OPTIONS END

data = data(languages)
data.collectXY(collect_languages) 
labels = list(set(data.Y[label_prediction]))

if use_cross_validation:
  kfold_accuracy = []
  kfold_precision = []
  kfold_recall = []
  kfold_f1score = []

  kf = KFold(n_splits=K_folds)

  i = 1
  for train_index, test_index in kf.split(data.X):
    X_train, X_test = np.array(data.X)[train_index], np.array(data.X)[test_index]
    Y_train, Y_test = np.array(data.Y[label_prediction])[train_index], np.array(data.Y[label_prediction])[test_index]

    classifier = svmClassifier(X_train, X_test, Y_train, Y_test, labels)
    classifier.classify()
    classifier.evaluate()
    classifier.printBasicEvaluation()

    kfold_accuracy.append(classifier.accuracy)
    kfold_precision.append(classifier.precision)
    kfold_recall.append(classifier.recall)
    kfold_f1score.append(classifier.f1score)

  basicFunctions.printEvaluation(       basicFunctions.avg(kfold_accuracy),
                                        basicFunctions.avg(kfold_precision),
                                        basicFunctions.avg(kfold_recall),
                                        basicFunctions.avg(kfold_f1score),
                                        str(K_folds) + "-Fold Cross Validation Evaluation"
  )

else:
  X_train, X_test, Y_train, Y_test = train_test_split(data.X, data.Y[label_prediction], test_size=0.20, random_state=42)
  classifier = svmClassifier(X_train, X_test, Y_train, Y_test, labels) #what do we want to classify? Y['age'] or Y['gender']
  classifier.classify()
  classifier.evaluate()
  classifier.printBasicEvaluation()


  

