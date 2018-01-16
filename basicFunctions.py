from collections import Counter
class basicFunctions:
  def avg(l):
    return sum(l) / len(l)

  def printEvaluation(accuracy, precision, recall, f1score, text):    
    print("~~~" + text + "~~~ \n")
    print("Accuracy:\t {}".format(round(accuracy, 3)))
    print("Precision:\t {}".format(round(precision, 3)))
    print("Recall:\t\t {}".format(round(recall, 3)))
    print("F1-Score:\t {}".format(round(f1score, 3)))
  
  def printClassEvaluation(precision, recall, f1score, label):
    print('{} \t {} \t\t {} \t\t {}'.format(
      label,
      round(precision, 3),
      round(recall, 3),
      round(f1score, 3)
    ))
  
  def getLabelDistribution(labels):
    print(Counter(labels))