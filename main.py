
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from svmClassifier import SVM
from bayesClassifier import Bayes
from kNeighborsClassifier import KNeighbors
from decisionTreeClassifier import DecisionTree

from data import data

from basicFunctions import basicFunctions


random.seed(3)

# Read arguments
parser = argparse.ArgumentParser(description='system parameters')
parser.add_argument('--method', type=str, default='svm', help='machine learning technique')
parser.add_argument('--data_method', type=int, default=1, help='how to divide the data') #all documents from 1 user in one string, or every document in one string
parser.add_argument('--predict_languages', type=str, default='e', help='predict languages: language name seperated with komma or first letter of the language (without komma)') # ONLY USE English and Spanish for AGE!
parser.add_argument('--predict_label', type=str, default='gender', help='predict age or gender')
parser.add_argument('--avoid_skewness', type=bool, default=False, help='how to train the dataset, without skewness in the data or with skewness')
parser.add_argument('--kfold', type=int, default=1, help='Amount of Ks for cross validation, if cross validation.')
args = parser.parse_args()

predict_languages = basicFunctions.getLanguages(args.predict_languages)

data = data(predict_languages)
data.collectXY(data_method=args.data_method) 

basicFunctions.printStandardText(args.method)

labels = list(set(data.Y[args.predict_label]))
basicFunctions.printLabelDistribution(data.Y[args.predict_label])

if len(labels) > 1: #otherwise, there is nothing to train
  if args.kfold > 1:
    kfold_accuracy = []
    kfold_precision = []
    kfold_recall = []
    kfold_f1score = []

    kf = KFold(n_splits=args.kfold)

    i = 1
    for train_index, test_index in kf.split(data.X):
      X_train, X_test = np.array(data.X)[train_index], np.array(data.X)[test_index]
      Y_train, Y_test = np.array(data.Y[args.predict_label])[train_index], np.array(data.Y[args.predict_label])[test_index]
      
      if args.avoid_skewness:
        X_train, Y_train = basicFunctions.getUnskewedSubset(X_train, Y_train)

      if args.method == 'bayes':
        classifier = Bayes(X_train, X_test, Y_train, Y_test, labels)
      elif args.method == 'svm':
        classifier = SVM(X_train, X_test, Y_train, Y_test, labels)
      elif args.method == 'knear':
        classifier = KNeighbors(X_train, X_test, Y_train, Y_test, labels)
      elif args.method == 'tree':
        classifier = DecisionTree(X_train, X_test, Y_train, Y_test, labels)
      classifier.classify()
      classifier.evaluate()
      #classifier.printBasicEvaluation()
      #classifier.printClassEvaluation()

      kfold_accuracy.append(classifier.accuracy)
      kfold_precision.append(classifier.precision)
      kfold_recall.append(classifier.recall)
      kfold_f1score.append(classifier.f1score)

    basicFunctions.printEvaluation(       basicFunctions.avg(kfold_accuracy),
                                          basicFunctions.avg(kfold_precision),
                                          basicFunctions.avg(kfold_recall),
                                          basicFunctions.avg(kfold_f1score),
                                          str(args.kfold) + "-Fold Cross Validation Evaluation"
    )

  else:
    X_train, X_test, Y_train, Y_test = train_test_split(data.X, data.Y[args.predict_label], test_size=0.20, random_state=42)
    
    if args.avoid_skewness:
      X_train, Y_train = basicFunctions.getUnskewedSubset(X_train, Y_train)

    # for label in trainingDistribution:
    #   trainingDistribution[label]
    if args.method == 'bayes':
      classifier = Bayes(X_train, X_test, Y_train, Y_test, labels) 
    elif args.method == 'svm':
      classifier = SVM(X_train, X_test, Y_train, Y_test, labels) 
    elif args.method == 'knear':
      classifier = KNeighbors(X_train, X_test, Y_train, Y_test, labels)
    elif args.method == 'tree':
      classifier = DecisionTree(X_train, X_test, Y_train, Y_test, labels)
    classifier.classify()
    classifier.evaluate()
    classifier.printBasicEvaluation()
    classifier.printClassEvaluation()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))