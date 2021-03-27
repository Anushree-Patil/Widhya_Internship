#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('Z:\widhya/bitcoin_dataset.csv')


# In[104]:


df.head()


# In[105]:


df.iloc[1023,1
       ]


# In[106]:


correlation =df.corr(method='pearson')
# print(correlation)
correlation_matrix = correlation.abs()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
tri_df = correlation_matrix.mask(mask)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > .99)]
import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(correlation, 
            xticklabels=correlation.columns,
            yticklabels=correlation.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=1)


# In[107]:


df = df.fillna(df.mean())
x.shape


# In[108]:


y.shape


# In[110]:


y= df['btc_market_price']
columns= ['btc_market_cap', 'btc_n_transactions','btc_miners_revenue','btc_cost_per_transaction','btc_hash_rate','btc_difficulty','btc_cost_per_transaction_percent']

x= df[columns]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state =0)
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
sklearn.metrics.mean_squared_error(y_test, y_pred)


# In[ ]:





# In[ ]:




