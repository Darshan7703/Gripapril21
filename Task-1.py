#!/usr/bin/env python
# coding: utf-8

# # Data Science & Buisness Analytics Internship

# ## Author : Darshan Panchal

# ### Task : 1 :   Predection Using Supervised ML

# ## Problem : Predict the percentage of an student based on the no. of study hours.

# ### Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("http://bit.ly/w-data")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[9]:


data.describe()


# In[10]:


data.info()


# # Data Visualization:

# In[11]:


plt.boxplot(data)
plt.show()


# In[12]:


plt.xlabel("Hours",fontsize = 14)
plt.ylabel("Scores",fontsize = 14)
plt.title("Study Hours vs Scores")
plt.scatter(data.Hours,data.Scores,color='green',marker ='*')
plt.show()


# In[15]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values
X


# In[17]:


Y


# ## Training the dataset

# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size =0.2)


# In[19]:


print("Shape of X_train : ",X_train.shape)
print("Shape of Y_train : ",Y_train.shape)
print("Test of X_train : ",X_test.shape)
print("Test of Y_train : ",Y_test.shape)


# In[20]:


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()


# In[21]:


linearRegression.fit(X_train,Y_train)


# In[22]:


print("B0 =",linearRegression.intercept_,"\nB1 =",linearRegression.coef_)


# In[23]:


Y0 = linearRegression.intercept_ + linearRegression.coef_ * X_train


# ## Plotting the Regression Line

# In[24]:


plt.scatter(X_train,Y_train,color='red',marker='+')
plt.plot(X_train,Y0,color='black')
plt.xlabel("Hours",fontsize=14)
plt.ylabel("Scores",fontsize=14)
plt.title("Regression line(Train set)",fontsize=10)
plt.show()


# In[25]:


Y_pred = linearRegression.predict(X_test)
print(Y_pred)


# In[26]:


Y_test


# In[27]:


plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='black',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=10)
plt.show()


# In[28]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# ## Calculating the metrices

# In[29]:


from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)


# In[30]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[31]:


MSE = metrics.mean_squared_error(Y_test,Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error      = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error     = ",Abs_E)


# # Predicting the score for 9.25 hours¶

# In[32]:


Prediction_score = linearRegression.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)


# # Thank You !!

# In[ ]:




