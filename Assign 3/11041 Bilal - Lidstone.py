#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.preprocessing import MinMaxScaler
trainDTar = train.drop(columns=['target'])
scaler = MinMaxScaler()
print(scaler.fit(trainDTar))
newtrain = scaler.transform(trainDTar)
lstCol = list(trainDTar.columns)
newtrainDF = pd.DataFrame(newtrain, columns=lstCol)
newtrainDF


# In[61]:


X = newtrainDF
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=39)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[83]:


from sklearn.naive_bayes import MultinomialNB 
mnb_clf = MultinomialNB()
res = mnb_clf.fit(X_train, y_train)
mnbPred = mnb_clf.predict(X_test)
mnbAcc = metrics.accuracy_score(y_test, mnbPred)
print (mnbAcc*100)


# In[110]:


nav_clf = MultinomialNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=90)
print('Naive Bayes Scores: ',nav_scores*100)
nav_mean = nav_scores.mean()
print('Naive Bayes Mean Score: ',nav_mean*100)


# In[111]:


newCSV = test[['id']]
newCSV


# In[112]:


prd = test.drop(columns=['id'])
prd.head(1)


# In[113]:


testPRD = mnb_clf.predict(prd)
print(testPRD)


# In[114]:


newCSV['target'] = testPRD
newCSV.head()


# In[115]:


newCSV.to_csv('output.csv', index=False)


# In[ ]:




