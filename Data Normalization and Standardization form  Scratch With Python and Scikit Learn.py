
# coding: utf-8

# In[37]:


#Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[38]:


#Load data
df= pd.read_csv(r"D:\Hackathon\Big mart sales problem - Analytics Vidhya\train_v9rqX0R.csv")

df.head()


# In[39]:


data= df[['Item_Weight','Item_MRP']]
data


# In[40]:


data.plot.kde()


# <h2>Normalization using Python </h2>

# In[41]:


# copy the data
df_max_scaled = data.copy()

# apply normalization techniques
for column in df_max_scaled.columns:
    
    df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

# view normalized data
display(df_max_scaled)


# In[42]:


df_max_scaled.plot.hist()


# <h3>Normalization The min-max feature scaling</h3>
# 
# The min-max approach (often called normalization) rescales the feature to a hard and fast range of [0,1] by subtracting the minimum value of the feature then dividing by the range. We can apply the min-max scaling in Pandas using the .min() and .max() methods.

# In[43]:


for column in df_max_scaled.columns:
    #print(column)
    df_max_scaled[column] = (df_max_scaled[column]-df_max_scaled[column].min()) / (df_max_scaled[column].abs().max()-df_max_scaled[column].min())

# view normalized data
display(df_max_scaled)


# In[44]:


df_max_scaled.plot.kde()


# <h2>Normalization using Scikit Learn </h2>

# In[45]:


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit_transform(data)
norm


# In[46]:


import seaborn as sns
sns.kdeplot(data=norm)


# <h3> Standardization Using The z-score method from Python</h3>
# 
# The z-score method (often called standardization) transforms the info into distribution with a mean of 0 and a typical deviation of 1. Each standardized value is computed by subtracting the mean of the corresponding feature then dividing by the quality deviation.

# In[47]:


for column in df_max_scaled.columns:
    
    df_max_scaled[column] = (df_max_scaled[column]-df_max_scaled[column].mean()) / df_max_scaled[column].std()

# view normalized data
display(df_max_scaled)


# In[48]:


df_max_scaled.plot.kde()


# <h3> Standardization using Scikit_learn</h3>

# In[50]:


from sklearn.preprocessing import StandardScaler

trans_data= StandardScaler().fit_transform(data)


# In[51]:


trans_data


# In[53]:


import seaborn as sns
sns.kdeplot(data=trans_data)

