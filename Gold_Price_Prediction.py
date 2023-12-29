#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MYoussef885/Gold_Price_Prediction/blob/main/Gold_Price_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Importing the Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data collection and processing

# In[4]:


# loading the csv data to a pandas dataframe
gold_data = pd.read_csv('/content/gold_price_data.csv')


# In[5]:


# print first five rows in the dataframe
gold_data.head()


# In[6]:


# print the last five rows of the dataframe
gold_data.tail()


# In[7]:


# number of rows and columns
gold_data.shape


# In[8]:


# getting some information about data
gold_data.info()


# In[9]:


# checking number of missing values
gold_data.isnull().sum()


# In[10]:


# statistical measures of data
gold_data.describe()


# Correlation:
# 1. Positive Correlation
# 2. Negative Correlation

# In[11]:


correlation = gold_data.corr


# In[ ]:


# constructing a heatmap tp understand the correlation
#plt.figure(figsize = (8,8))
#sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap ='Blues')


# In[15]:


# correlation values of GLD
#print(correlation['GLD'])


# In[17]:


# checking the distribution of the GLD price
sns.displot(gold_data['GLD'], color='green')


# Splitting the features and target

# In[18]:


X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']


# In[19]:


print(X)


# In[20]:


print(Y)


# Splitting into Training and Test Data

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# Model Training:
# Random Forest Regressor

# In[22]:


regressor = RandomForestRegressor(n_estimators=100)


# In[23]:


# training the model
regressor.fit(X_train, Y_train)


# In[24]:


# prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[25]:


print(test_data_prediction)


# In[26]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print('R squared error:', error_score)


# Compare the actual values and predicted values in a plot

# In[27]:


Y_test = list(Y_test)


# In[28]:


plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




