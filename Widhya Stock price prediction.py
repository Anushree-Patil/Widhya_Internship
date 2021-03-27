#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install quandl')


# In[98]:


import os, math
import pandas as pd
import numpy as np
import quandl
import time
import sklearn
from sklearn import preprocessing, svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn. model_selection import cross_validate
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = "_anSK6QSYPCxXvxN4Zjs"
df = quandl.get("EOD/AAPL")


# In[99]:


df.head()


# In[100]:


df['HL_PCT'] = (df['Adj_High']-df['Adj_Low'])/df['Adj_Close'] *100.0
df['PCT_change'] = (df['Adj_Close']-df['Adj_Open'])/df['Adj_Open'] *100.0
df.head()


# In[101]:


df = df.loc[:,['Adj_Close','PCT_change','HL_PCT','Adj_Volume']]
df.head()


# In[102]:


forecast_col = 'Adj_Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))


# In[103]:


df.isnull().sum()


# In[104]:


df['label'] = df[forecast_col].shift(-forecast_out)


# In[105]:


df.head()


# In[106]:


df.dropna(inplace=True)
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])


# In[107]:


X = preprocessing.scale(X)
y = np.array(df['label'])


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[109]:


model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
sklearn.metrics.mean_squared_error(y_test, y_pred)


# In[ ]:




