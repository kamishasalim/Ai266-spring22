#!/usr/bin/env python
# coding: utf-8

# In[236]:


import pandas as pd
from sklearn.metrics import accuracy_score # For Checking Accuracy
from sklearn.model_selection import train_test_split # Splitting Data For Train Test
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB # For Multinomial Naive Bayes Model
from sklearn.model_selection import cross_val_score # For Cross Validation
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/Users/bilal/Desktop/train.csv')
test = pd.read_csv('C:/Users/bilal/Desktop/test.csv')
print(train.shape)
print(test.shape)
del train['id']
del train['f_27']
del test['f_27']
train.head()


# In[238]:


from sklearn.preprocessing import MinMaxScaler
trainDTar = train.drop(columns=['target'])
scaler = MinMaxScaler()
print(scaler.fit(trainDTar))
newtrain = scaler.transform(trainDTar)
lstCol = list(trainDTar.columns)
newtrainDF = pd.DataFrame(newtrain, columns=lstCol)
newtrainDF


# In[239]:


X = newtrainDF
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[240]:


from sklearn.naive_bayes import MultinomialNB 
mnb_clf = MultinomialNB()
res = mnb_clf.fit(X_train, y_train)
mnbPred = mnb_clf.predict(X_test)
mnbAcc = metrics.accuracy_score(y_test, mnbPred)
print (mnbAcc*100)


# In[241]:


nav_clf = MultinomialNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=10)
print('Naive Bayes Scores: ',nav_scores*100)
nav_mean = nav_scores.mean()
print('Naive Bayes Mean Score: ',nav_mean*100)


# In[242]:


newCSV = test[['id']]
newCSV


# In[244]:


prd = test.drop(columns=['id'])
prd.head(3)


# In[246]:


testPRD = mnb_clf.predict(prd)
testPRD


# In[247]:


newCSV['target'] = testPRD
newCSV


# In[248]:


newCSV.to_csv('output.csv', index=False)


# In[ ]:




