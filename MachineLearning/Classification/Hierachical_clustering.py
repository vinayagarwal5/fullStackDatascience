# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:07:45 2026

@author: vinay agrawal
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:30:18 2026

@author: vinay agrawal
"""
#unsupervised K-Means clustering.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\data\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X, method= 'ward'))



plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucleadean Distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, metric= 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
