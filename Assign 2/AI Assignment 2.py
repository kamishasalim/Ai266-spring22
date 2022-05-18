#!/usr/bin/env python
# coding: utf-8

#Importing Libraries
import pandas as pd
import numpy as np

#Reading Test Data
testDf = pd.read_csv('C:/Users/ucom/Desktop/test.csv')

#Getting ID and removing everything else
idDf = testDf[['id']];

#Creating target column as per Kaggle's specifications 
idDf.insert(1,"target",0.0000000000);

#Generating random values
idDf['target'] = np.random.rand(700000,1);

#Writing back to a csv file
idDf.to_csv('output.csv');
print(idDf);