from collections import Counter
import random
import numpy as np

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class basicFunctions:
  def avg(l):
    return sum(l) / len(l)

  def keyCounter(list):
    return Counter(list)

  def printStandardText(method):
    print('#'*67)
    print('{} \t LFD Output \t\t\t\t {}'.format('#'*10, '#'*10))
    print('{} \t Machine Learning Method: {} \t\t {}'.format('#'*10, method, '#'*10))
    print('#'*67)

  def printEvaluation(accuracy, precision, recall, f1score, text):    
    print("~~~" + text + "~~~ \n")
    print("Accuracy:\t {}".format(round(accuracy, 3)))
    print("Precision:\t {}".format(round(precision, 3)))
    print("Recall:\t\t {}".format(round(recall, 3)))
    print("F1-Score:\t {}".format(round(f1score, 3)))
  
  def printClassEvaluation(Y_test, Y_predicted, labels):

    print("\n~~~ Class Evaluation ~~~ \n")
    print("Class \t Precision \t Recall \t F-score")

    for label in labels:
      accuracy, precision, recall, f1score = basicFunctions.getMetrics(Y_test, Y_predicted, [label])
      print('{} \t {} \t\t {} \t\t {}'.format(
        label,
        round(precision, 3),
        round(recall, 3),
        round(f1score, 3)
      ))
  def printLabelDistribution(labels):
    label_distribution = basicFunctions.keyCounter(labels)

    print('~~~Label Distribution~~~')
    for label in label_distribution:
      print('{} \t {}'.format(label, label_distribution[label]))
    print()

  def getMetrics(Y_test, Y_predicted, labels):
    accuracy = np.mean(Y_predicted == Y_test)
    already_set = False
    clean_labels = [] #without errors
    if len(labels) == 1:
      if labels[0] not in Y_predicted:
        precision = 0.0
        recall = 0.0
        f1score = 0.0
        already_set = True
      clean_labels.append(labels[0])
    else:
      for label in labels:
        if label in Y_predicted:
          clean_labels.append(label)

    if already_set == False:
      precision = sklearn.metrics.precision_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
      recall = sklearn.metrics.recall_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
      f1score = sklearn.metrics.f1_score(Y_test, Y_predicted, average="macro", labels=clean_labels)

    return accuracy, precision, recall, f1score

  def getLanguages(argument_languages):
    possible_languages = ['dutch', 'english', 'spanish', 'italian']
    if argument_languages == 'all':
      predict_languages = possible_languages

    predict_languages = argument_languages.split(',')

    new_format = []
    if len(predict_languages) == 1:
      if predict_languages[0] not in possible_languages:
        for letter in predict_languages[0]:
          for possible_language in possible_languages:
            if letter == possible_language[0]:
              new_format.append(possible_language)
        predict_languages = new_format
    return predict_languages

  def getUnskewedSubset(X_train, Y_train):
    dataDistribution = basicFunctions.keyCounter(Y_train)
    lowestAmount = 0
    for label in dataDistribution:
      if dataDistribution[label] < lowestAmount or lowestAmount == 0:
        lowestAmount = dataDistribution[label]
    keyDict = {}
    for i, label in enumerate(Y_train):
      if label not in keyDict:
        keyDict[label] = [i]
      else:
        keyDict[label].append(i)

    new_X_train = []
    new_Y_train = []
    all_keys = []
    newDict = {}
    for label in keyDict: 
      newDict[label] = random.sample(keyDict[label], lowestAmount)
      all_keys += newDict[label]
      for i in newDict[label]:

        new_X_train.append(X_train[i])
        new_Y_train.append(Y_train[i])

    return new_X_train, new_Y_train