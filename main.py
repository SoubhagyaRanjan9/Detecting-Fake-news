import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv('news.csv')
df.head()
df.shape

labels=df.label
labels.head()

#Split the dataset into training and testing sets

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=0)
print(len(x_train),len(x_test))

#Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#predict on the test set from the TfidfVectorizer 

pas=PassiveAggressiveClassifier(max_iter=40)
pas.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred=pas.predict(tfidf_test)
from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

#calculating confusion matrix
cm=confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])
print('confusion Matrix is :',cm,labels,sep='\n')

#So this model have 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives.