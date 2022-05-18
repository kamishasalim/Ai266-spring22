#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np

testDf = pd.read_csv('C:/Users/ucom/Desktop/test.csv')
idDf = testDf[['id']];
idDf.insert(1,"target",0.0000000000);
idDf['target'] = np.random.rand(700000,1);
idDf.to_csv('output.csv');
print(idDf);


# In[ ]:




