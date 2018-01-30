# Learning From Data
Run with the following command for our best system (only on training for now, as the test set is not available yet)
```
python3 main.py --data_folder=data/gender_age_prediction/ --predict_label=gender --predict_languages=esid --method=svm --k=5 --print_details=5
```
# SGD sklearn (random_state=42, max_iter=50, tol=None, loss='hinge')
**Languages:  eng, es (no dutch & ita because they have no age label)
Predict:    age**

customTokenizer, words 1-3, chars 2-5
Accuracy:   68,2
F-score:    64,3

tweetIdentity, words 1-3, chars 2-5
Accuracy:   68,2
F-score:    64,3

tweetIdentity + porterstemmer, words 1-2, chars 3-5
Accuracy:   69
F-score:    64,6

**tweetIdentity + porterstemmer, words 1-5, chars 3-5**
Accuracy:	 0.692
Precision:	 0.693
Recall:		 0.622
F1-Score:	 0.648

**Languages:  eng, es, dutch, ita
Predict:    gender**
Accuracy:	 0.74
Precision:	 0.741
Recall:		 0.74
F1-Score:	 0.739

# SVC sklearn (kernel='linear', decision_function_shape='ovr')
Languages:  eng, es
Predict:    age
Accuracy:   68
F-score:    64,5
