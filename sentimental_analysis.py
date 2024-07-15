

pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d kazanova/sentiment140

from zipfile import ZipFile
dataset = "sentiment140.zip"

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print("Done")

import numpy
import pandas as pd
#re - regular expression lib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')


twitter_data.shape


twitter_data.head()



column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv',names=column_names, encoding='ISO-8859-1')

twitter_data.shape

twitter_data.head()


twitter_data.isnull().sum()


twitter_data['target'].value_counts()


twitter_data.replace({'target':{4:1}},inplace=True)


twitter_data['target'].value_counts()


port_stem = PorterStemmer()

def stemming(content):
#content are tweets
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  #'[^a-zA-Z]' this means we are removing all the other things from tweets which are not alphabets like nums.special charcter
  stemmed_content = stemmed_content.lower()
  #making all the words in lower case because being upper case wil not contri anything in our analysis
  stemmed_content = stemmed_content.split()
  #here we are separating all the words of the tweet and then making the list using split
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  #here port_stem is use for making root words
  stemmed_content = ' '.join(stemmed_content)
  #here we are combining the words again after above processing has been done
  return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

twitter_data.head()

print(twitter_data['stemmed_content'])


X=twitter_data['stemmed_content'].values
Y=twitter_data['target'].values


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


print(X.shape,X_train.shape,X_test.shape)



vectorizer= TfidfVectorizer()

X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

print(X_train)


model = LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)

print(training_data_accuracy)


X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,X_test_prediction)

print(testing_data_accuracy)


import pickle

filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))



loaded_model=pickle.load(open('trained_model.sav','rb'))


X_new = X_test[200]
print(Y_test[200])

predictions=loaded_model.predict(X_new)
print(predictions)

if (predictions[0]==0):
  print('negative')
else:
  print('positive')

X_new = X_test[6]
print(Y_test[6])

predictions=loaded_model.predict(X_new)
print(predictions)

if (predictions[0]==0):
  print('negative')
else:
  print('positive')

