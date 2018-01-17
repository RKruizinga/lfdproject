
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from svmClassifier import svmClassifier
from bayesClassifier import bayesClassifier
from data import data
from basicFunctions import basicFunctions

# Read arguments
parser = argparse.ArgumentParser(description='system parameters')
parser.add_argument('--method', metavar='xx', type=str, default='svm', help='machine learning technique')
parser.add_argument('--data_method', metavar='xx', type=int, default=1, help='how to divide the data') #all documents from 1 user in one string, or every document in one string
parser.add_argument('--predict_languages', metavar='xx', type=str, default='es', help='predict languages: language name seperated with komma or first letter of the language (without komma)') # ONLY USE English and Spanish for AGE!
parser.add_argument('--predict_label', type=str, default='age', help='predict age or gender')
parser.add_argument('--kfold', type=int, default=1, help='Amount of Ks for cross validation, if cross validation.')
args = parser.parse_args()

predict_languages = basicFunctions.getLanguages(args.predict_languages)

data = data(predict_languages)
data.collectXY(data_method=args.data_method) 

labels = list(set(data.Y[args.predict_label]))

basicFunctions.printLabelDistribution(data.Y[args.predict_label])

if len(labels) > 1: #otherwise, there is nothing to train for
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
      if args.method == 'bayes':
        classifier = bayesClassifier(X_train, X_test, Y_train, Y_test, labels)
      if args.method == 'svm':
        classifier = svmClassifier(X_train, X_test, Y_train, Y_test, labels)
      classifier.classify()
      classifier.evaluate()
      #classifier.printBasicEvaluation()

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
    if args.method == 'bayes':
      classifier = bayesClassifier(X_train, X_test, Y_train, Y_test, labels) #what do we want to classify? Y['age'] or Y['gender']
    elif args.method == 'svm':
      classifier = svmClassifier(X_train, X_test, Y_train, Y_test, labels) #what do we want to classify? Y['age'] or Y['gender']
    classifier.classify()
    classifier.evaluate()
    classifier.printBasicEvaluation()
    classifier.printClassEvaluation()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))