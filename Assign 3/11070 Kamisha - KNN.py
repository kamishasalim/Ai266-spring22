#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import pickle 
import numpy as np
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/ucom/Desktop/train.csv")
df.shape
del df['id']
del df['f_27'] #deleting the irrelevent column
df.head(6)


# In[7]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
X = df.drop(columns=['target'])
y = df['target']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[17]:


newdf = df[["f_22","f_25","f_19","f_24","f_21","f_26","target"]]
newdf.head(6)


# In[20]:


XTT = newdf.drop(columns=['target'])
yTT = newdf['target']
X_train, X_test, y_train, y_test = train_test_split(XTT, yTT, test_size=0.2)
modelKNN = KNeighborsClassifier(n_neighbors=16)
resultKNN = modelKNN.fit(X_train, y_train)
prediction_test = modelKNN.predict(X_test)
accuracyKNN = metrics.accuracy_score(y_test, prediction_test)
print("KNN MODEL ACCURACY: ", accuracyKNN)


# In[22]:


from sklearn.model_selection import cross_val_score
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=5)
knn_mean = knn_scores.mean()
print("KNN MODEL MEAN: ",knn_mean)


# In[23]:


dftest = pd.read_csv("C:/Users/ucom/Desktop/test.csv")
dftest.head(6)


# In[25]:


newtest = dftest[["f_22","f_25","f_19","f_24","f_21","f_26"]]
newtest.head(6)


# In[26]:


newCSV = dftest[['id']]
newCSV


# In[27]:


predictionOnTest = modelKNN.predict(newtest)
predictionOnTest


# In[28]:


newCSV['target'] = predictionOnTest 
newCSV


# In[29]:


newCSV.to_csv('Output.csv')

