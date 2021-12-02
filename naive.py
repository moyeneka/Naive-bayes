import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# read the train and test dataset
train_Test_data = pd.read_csv('spam_ham_dataset.csv')

print(train_Test_data.groupby('label').describe())

# shape of the dataset
print('Shape of training data :',train_Test_data.shape) 
train_Test_data['spam'] = train_Test_data['label']   .apply(lambda x: 1 if x=='spam' else 0)
print(train_Test_data.head())

train_X, test_X, train_Y, test_Y = train_test_split(train_Test_data.text,train_Test_data.spam,test_size=0.25)



































































