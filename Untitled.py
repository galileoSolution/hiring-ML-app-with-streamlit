#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[3]:


data   = pd.read_csv("hiring.csv")


# In[12]:


data


# In[5]:


data.isnull().sum()


# In[13]:


data.experience  = data.experience.fillna(data.experience.mode())


# In[16]:


data["test_score(out of 10)"]  = data["test_score(out of 10)"].fillna(data["test_score(out of 10)"].mean().round())


# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


lb  = LabelEncoder()


# In[22]:


data.experience = lb.fit_transform(data.experience)


# In[25]:


data.dtypes


# In[27]:


data.columns = ['experience', 'test_score', 'interview_score',
       'salary($)']


# In[29]:


X = data.drop('salary($)', axis  = 1)
y  = data["salary($)"]


# In[31]:


model = LinearRegression()
model.fit(X,y)


# In[32]:


import pickle


# In[33]:


pickle.dump(model,open('model.pkl','wb'))


# In[ ]:





# In[34]:




# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




