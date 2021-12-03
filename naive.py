from typing import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay

# read the train and test dataset
train_Test_data = pd.read_csv('spam_ham_dataset.csv')

print(train_Test_data.groupby('label').describe())

# shape of the dataset
print('Shape of training data :',train_Test_data.shape) 
train_Test_data['spam'] = train_Test_data['label']   .apply(lambda x: 1 if x=='spam' else 0)
print(train_Test_data.head())

train_X, test_X, train_Y, test_Y = train_test_split(train_Test_data.text,train_Test_data.spam,test_size=0.25)
vec = CountVectorizer()
train_count_X = vec.fit_transform(train_X.values)
train_count_X.toarray()[:3]

model_Multi= MultinomialNB()
model_Multi.fit(train_count_X,train_Y)

test_count_X = vec.transform(test_X)
print("score Test set = ",model_Multi.score(train_count_X,train_Y))

print("score Training set = ",model_Multi.score(test_count_X,test_Y))

svc_disp = RocCurveDisplay.from_estimator(model_Multi, test_count_X, test_Y)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(model_Multi, test_count_X, test_Y, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
#plt.plot(test_count_X,test_Y)
#X train = text
#test set = spam or not



































































