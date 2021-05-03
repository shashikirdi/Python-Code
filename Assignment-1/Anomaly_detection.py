#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime

#Input data - crypto currency prices in dollars
data = [7845, 778, 942, 143, 0.75, 7956, 810, 976, 146, 0.76, 8215, 825, 1002, 152,
0.78, 8542, 847, 1038, 157, 0.78, 8150, 100587, 807, 1015, 150, 0.72, 8386,
884, 101964, 1085, 138, 0.82, 8219, 827, 995, 158, 0.82, 7500, 745, 948,
135, 0.67, 9257, 901, 120967, 1154, 148, 0.72, 8553, 811, 1218, 175, 0.84]

# Find data outside of the price range (0 - 20000 dollar)
corrupt_data = []
for i in range(0, len(data)):
    if (data[i] < 0.0 or data[i] > 20000.0):
        corrupt_data.append(data[i])

# print min, max and corrupted data in the data set
print("Minimum price in the data set:" + str(min(data)))
print("Maximum price in the data set:" + str(max(data)))
print("Corrupted data (>0 and <20,000 dollar):" + str(corrupt_data) )
        

#kmeans clustering = 6
x = np.array(data)
kmeans6 = KMeans(n_clusters=6)
y_kmeans6 = kmeans6.fit_predict(x.reshape(-1,1))
x_axis = [datetime.datetime.now() + datetime.timedelta(hours=i) for i in range(len(x))]

print("Kmeans [clustering = 6] result:" + str(y_kmeans6))
plt.scatter(x_axis, x)
plt.xlabel("Time")
plt.ylabel("Crypto currency price")
plt.show()

plt.scatter(y_kmeans6, x)
plt.xlabel("cluster index")
plt.ylabel("Crypto currency price")
plt.show()

#Kmeans clustering = 4
x = np.array(data)
kmeans6 = KMeans(n_clusters=4)
y_kmeans6 = kmeans6.fit_predict(x.reshape(-1,1))
x_axis = [datetime.datetime.now() + datetime.timedelta(hours=i) for i in range(len(x))]

print("Kmeans [clustering = 4] result:" + str(y_kmeans6))
plt.scatter(x_axis, x)
plt.xlabel("Time")
plt.ylabel("Crypto currency price")
plt.show()

plt.scatter(y_kmeans6, x)
plt.xlabel("cluster index")
plt.ylabel("Crypto currency price")
plt.show()

print(" From the plot for kmeans (clustering = 4), we could still detect the same corrupted prices")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




