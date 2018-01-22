import os
import xml.etree.ElementTree as ET

import re
class data:
  documents = {} #documents per language and per user
  labels = {}

  X = []
  Y = { 'age': [],
        'gender': [],
        'language': [] }

  def __init__(self, folders):
    self.readFiles(folders)
    self.languages = folders #get all languages wanted

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
                # child.text = child.text.strip('\t') #strip all tabs
          
                # child.text = child.text.replace('”', '') #strip all tabs
                # child.text = child.text.replace('“', '') #strip all tabs 
                # child.text = child.text.replace('"', '') #strip all tabs   

                child.text = re.sub(r'http:\/\/t.co\S*', 'URL', child.text)
                child.text = re.sub(r'https:\/\/t.co\S*', 'URL', child.text)
                child.text = re.sub(r'\d+\-\d+\-\d+', 'DATE', child.text)
                child.text = re.sub(r'[0-2][0-9]{3}', 'YEAR', child.text)

                # #child.text = re.sub(r'\d', '#', child.text) #replace each number with a char

                #child.text = re.sub(r'(.)\1{3,}', r'\1\1\1', child.text)
                #print(child.text)
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

  def collectXY(self, data_method = 1):
    if data_method == 1:
      for language in self.languages:
        for user in self.documents[language]:
          for document in self.documents[language][user]:
            self.X.append(document)
            self.Y['gender'].append(self.labels[language][user][0])
            self.Y['age'].append(self.labels[language][user][1])
            self.Y['language'].append(language)

    elif data_method == 2:
      for language in self.languages:
        for user in self.documents[language]:
          self.X.append('. '.join(self.documents[language][user]))
          self.Y['gender'].append(self.labels[language][user][0])
          self.Y['age'].append(self.labels[language][user][1])
          self.Y['language'].append(language)

  

