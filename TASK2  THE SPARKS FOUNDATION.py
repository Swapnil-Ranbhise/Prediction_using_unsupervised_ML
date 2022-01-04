#!/usr/bin/env python
# coding: utf-8

# # Prediction using Unsupervised ML.
# 
# From the given 'iris' dataset, predict the optimum number of clusters and represent it visually.

# # language
# Python

# # Author
# Swapnil Ranbhise

# In[14]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df = pd.read_csv("Iris.csv")
df.head()


# In[16]:


# findin the optimum number of cluster for kmeans classification
x = df.iloc[:,[0,1,2,3]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++",
                   max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()


# In[17]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(x)
y_predicted


# In[18]:


km.cluster_centers_


# In[20]:


plt.scatter(x[y_predicted == 0, 0], x[y_predicted == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_predicted == 1, 0], x[y_predicted == 1, 1], 
            s = 100, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(x[y_predicted == 2, 0], x[y_predicted == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',label='centroid')
plt.legend()


# In[21]:


scaler = MinMaxScaler()

scaler.fit(x)
x = scaler.transform(x)


# In[22]:


x


# In[ ]:




