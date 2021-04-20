#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Internship

# ## Author :- Darshan Panchal

# ## Task :- 2>> Prediction Using Unsupervised ML

# ### Problem :-  From the given ‘Iris’ dataset, predict the optimum number of clustersand represent it visually.

# In[21]:


#Dataset : https://bit.ly/3kXTdox


# ## Importing Required Libraries.

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


# In[14]:


from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


df1 = pd.read_csv("Iris.csv")
df1.head()


# In[19]:


df1["Species"].value_counts()


# ## Scatter Plot:-
# 
#                         A scatter plot (aka scatter chart, scatter graph) uses dots to represent values for two different numeric variables. The position of each dot on the horizontal and vertical axis indicates values for an individual data point. Scatter plots are used to observe relationships between variables.

# In[20]:


df1.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm",color="purple")


# ## Joint Plot:-
# 
#                    Seaborn’s Jointplot displays a relationship between 2 variables (bivariate) as well as 1D profiles (univariate) in the margins.
#                    Currently, jointplot wraps JointGrid with the following options for kind:
# 
#     -->> scatter - reg - resid - kde - hex
#     -->> ‘reg’ plots a linear regression line.

# In[25]:


with sns.axes_style('white'):
    sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df1, kind='kde',color="#900C3F");


# In[24]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df1,color="#357107")


# ## Box Plot:-
#                     Understanding the anatomy of a boxplot by comparing a boxplot against the probability density function for a normal distribution.

# In[26]:


sns.boxplot(x="Species", y="PetalLengthCm", data=df1)


# ## Violin Plot:--
#                     A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared.
#                     Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.

# In[27]:


sns.violinplot(x="Species", y="PetalLengthCm", data=df1)


# ## Rel Plot:--
#                   This function provides access to several different axes-level functions that show the relationship between two variables with semantic mappings of subsets. The kind parameter selects the underlying axes-level function to use:
#                   scatterplot() (with kind="scatter"; the default)
# 
#                   lineplot() (with kind="line")

# In[28]:


sns.relplot(
    data=df1,
    x="SepalLengthCm", y="SepalWidthCm", col="Species",
    hue="Species",
)


# # Api Abstraction Across Visualization :>>
#             There is no universally best way to visualize data. Different questions are best answered by different plots. Seaborn makes it easy to switch between different visual representations by using a consistent dataset-oriented API.
#             The function relplot() is named that way because it is designed to visualize many different statistical relationships. While scatter plots are often effective, relationships where one variable represents a measure of time are better represented by a line. The relplot() function has a convenient kind parameter that lets you easily switch to this alternate representation:

# In[29]:


sns.relplot(
    data=df1, kind="line",
    x="SepalLengthCm", y="SepalWidthCm", col="Species",
    hue="Species",
    facet_kws=dict(sharex=False),
)


# # Statistical estimation:>>
# 
#                 Many seaborn functions will automatically perform the statistical estimation that is necessary to answer the task related questions.

# In[30]:


sns.pairplot(df1.drop("Id", axis=1), hue="Species")


# In[31]:


from pandas.plotting import andrews_curves
andrews_curves(df1.drop("Id", axis=1), "Species")


# In[32]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(df1.drop("Id", axis=1), "Species")


# In[33]:


le=LabelEncoder()
df1['Species']=le.fit_transform(df1['Species'])
df1['Species'].value_counts()


# In[34]:


df1.corr()


# In[35]:


fig=plt.figure(figsize=(8,6))
sns.heatmap(df1.corr(),linewidths=1,annot=True)


# In[36]:


x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
plt.plot(range(1, 11), wcss, c="purple")
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[37]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[38]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend();


# # Thankyou!!!

# In[ ]:




