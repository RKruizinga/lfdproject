
from sklearn.model_selection import train_test_split

from svmClassifier import svmClassifier
from bayesClassifier import bayesClassifier
from data import data
from basicFunctions import basicFunctions

### OPTIONS BEGIN
method = 'bayes'

languages = ['dutch', 'english', 'spanish', 'italian']

collect_languages = 'all' #which language of documents
label_prediction = 'age' #predict age or gender

use_cross_validation = False #option to toggle between cross validation and simple validation
K_folds = 5
### OPTIONS END

data = data(languages)
data.collectXY(collect_languages) 
labels = list(set(data.Y[label_prediction]))

basicFunctions.getLabelDistribution(data.Y[label_prediction])

if len(labels) > 1: #otherwise, there is nothing to train for
  if method == 'svm':
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
        #classifier.printBasicEvaluation()

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
      classifier.printClassEvaluation()

  elif method == 'bayes':
    X_train, X_test, Y_train, Y_test = train_test_split(data.X, data.Y[label_prediction], test_size=0.20, random_state=42)
    classifier = bayesClassifier(X_train, X_test, Y_train, Y_test, labels) #what do we want to classify? Y['age'] or Y['gender']
    classifier.classify()
    classifier.evaluate()
    classifier.printBasicEvaluation()
    classifier.printClassEvaluation()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(collect_languages, label_prediction))
