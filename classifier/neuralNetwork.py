import numpy as np
import re
import os

import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
    
from sklearn.model_selection import train_test_split

from _function.basic import metrics, unskewedTrain
from _class.printer import Printer
from text.tokenizer import TextTokenizer
from _function.custom import writeWordEmbeddings, readWordEmbeddings

class NeuralNetwork:
  word_embeddings_file = 'data/word_embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
  word_embeddings_dim = 200
  word_embeddings_layer = None
  word_embeddings_index = {}

  labels = []
  labels_dict = {}
  labels_dict_rev = {}
  
  Y = []
  
  def __init__(self, data, show_fitting):
    self.data = data

    self.X = self.data.X
    self.labels = self.data.labels

    for i, label in enumerate(self.labels):
      self.labels_dict[label] = i
      self.labels_dict_rev[i] = label

    self.Y = []
    for label in self.data.Y:
      self.Y.append(self.labels_dict[label])
    
    self.show_fitting = show_fitting


  def tokenize(self):
    self.X_tokenized = TextTokenizer.tokenizeTweets(self.X) #all tweets!
    self.tokenizer = Tokenizer(split="|",)
    self.tokenizer.fit_on_texts(self.X_tokenized)
    self.sequences = self.tokenizer.texts_to_sequences(self.X_tokenized)
    self.X = pad_sequences(self.sequences)
    self.Y = to_categorical(self.Y)

  def classify(self, features, classifier=None):
    self.tokenize()

    train_development_split = self.data.amount_train 
    development_test_split = self.data.amount_train + self.data.amount_development

    self.X_train = self.X[:train_development_split]
    self.Y_train = self.Y[:train_development_split]

    self.X_development = self.X[train_development_split:development_test_split]
    self.Y_development = self.Y[train_development_split:development_test_split]

    
    self.X_test = self.X[development_test_split:]

    if self.data.avoid_skewness:
      Y_train = np.argmax(self.Y_train, axis=1)
      Y_train = [self.labels_dict_rev[int(i)] for i in list(Y_train)]
      
      self.X_train, self.Y_train = unskewedTrain(self.X_train, self.Y_train, Y_train)
      self.X_train = np.array(self.X_train)
      self.Y_train = np.array(self.Y_train)

    self.word_embeddings_layer, self.word_embeddings_index = readWordEmbeddings(self.data.languages, self.data.response_variable)
    if self.word_embeddings_layer == None:
      self.createWordEmbeddings()

    self.printDataInformation()

    ##CHANGE OPTIONS HERE
    self.model = Sequential()
    self.model.add(self.word_embeddings_layer)
    self.model.add(Dropout(0.2))
    self.model.add(LSTM(self.word_embeddings_dim))
    self.model.add(Dense(self.Y.shape[1], activation='sigmoid'))

    self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 
	
    # Train the model 
    self.printer = Printer('Model Fitting', self.show_fitting)
    self.model.fit(self.X_train, self.Y_train, epochs = 5, batch_size = 128, validation_split = 0.2)
    self.printer.duration()

  def evaluate(self):
    self.Y_development_predicted = self.model.predict(self.X_development)

    self.Y_development_predicted = np.argmax(self.Y_development_predicted, axis=1)
    self.Y_development_predicted = [self.labels_dict_rev[int(i)] for i in list(self.Y_development_predicted)]
    
    self.Y_development = np.argmax(self.Y_development, axis=1)
    self.Y_development = [self.labels_dict_rev[int(i)] for i in list(self.Y_development)]

    self.accuracy, self.precision, self.recall, self.f1score = metrics(self.Y_development, self.Y_development_predicted, self.labels)

  def printBasicEvaluation(self):    
    self.printer.evaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    self.printer.classEvaluation(self.Y_development, self.Y_development_predicted, self.labels)

  def printDataInformation(self):

    print('\n~~~Neural Network Distribution~~~\n')
    print('Found {} unique tokens.'.format(len(self.tokenizer.word_index)))
    print('Shape of data tensor: {}'.format(self.X.shape))
    print('Shape of label tensor: {}\n'.format(self.Y.shape))

    if len(self.word_embeddings_index) > 0:
      print('Found {} word vectors.'.format(len(self.word_embeddings_index)))

  def createWordEmbeddings(self):
    self.word_embeddings_index = {}
    f = open(self.word_embeddings_file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        self.word_embeddings_index[word] = coefs
    f.close()

    self.word_embeddings_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.word_embeddings_dim))
    for word, i in self.tokenizer.word_index.items():
        embedding_vector = self.word_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            self.word_embeddings_matrix[i] = embedding_vector

    self.word_embeddings_layer = Embedding(len(self.tokenizer.word_index) + 1, self.word_embeddings_dim, mask_zero = True, weights=[self.word_embeddings_matrix], trainable = True)

    writeWordEmbeddings(self.word_embeddings_layer, self.word_embeddings_index, self.data.languages, self.data.response_variable)

    

    
    


