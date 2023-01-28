#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import random
from numpy.linalg import matrix_power
import gurobipy as grb
import time

from ipynb.fs.full.MinMaxCorrelationClustering import exact
from ipynb.fs.full.MinMaxCorrelationClustering import cluster
from ipynb.fs.full.MinMaxCorrelationClustering import LocalObj
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLP
from ipynb.fs.full.MinMaxCorrelationClustering import PivotAlg
from ipynb.fs.full.MinMaxCorrelationClustering import DegreeDist
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLPonly
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLPneighbors


# In[2]:


#Read in the data
df = pd.read_csv('FilenameGraph.csv')

sources = list(df['NODE1'])
targets = list(df['NODE2'])


# In[3]:


#Construct the positive adjacency matrix
m = len(sources)
node_dict = {}
count = 0
names = []

for i in range(m):
    
    if sources[i] not in node_dict:
        node_dict.update({sources[i]: count})
        count = count + 1 
        names.append(sources[i])

    if targets[i] not in node_dict:
        node_dict.update({targets[i]: count})
        count = count + 1
        names.append(targets[i])
        
n = len(node_dict)
fb_adj_mx = np.zeros([n, n])

for i in range(m):
    u = node_dict[sources[i]]
    v = node_dict[targets[i]]
    fb_adj_mx[u][v] = 1
    fb_adj_mx[v][u] = 1
    
fb_adj_mx = fb_adj_mx + np.identity(n)


# In[4]:


#Read in the ground truth data 
df2 = pd.read_csv('FilenameCircles.csv')
circles = []
for k in df2.keys():
    circle = list(df2[k])
    clean_circ = []
    for j in range(len(circle)):
        if math.isnan(circle[j]):
            break
        else:
            clean_circ.append(int(circle[j]))
    circles.append(clean_circ)


# In[5]:


degrees = DegreeDist(fb_adj_mx)


# In[6]:


#Parameter sweep of radii in the rounding algorithm 
best_results = []
r1s = []
r2s = []
cluster_cts = []
obj_vals_equal_radii = []
equal_radii = []
print('Results for r1 = r2')
for i in range(21):
    for j in range(21-i-1):
        r1 = i/20
        r2 = r1 + j/20
        distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val = exact(fb_adj_mx, r1, r2)
        clustering, cluster_clock = cluster(distances, L_t_vals, neighborsR, neighborsR2, r1, r2)
        degrees = DegreeDist(fb_adj_mx)
        disag_vector, alg_obj_val, obj_vx = LocalObj(fb_adj_mx, clustering, degrees, math.inf)
        best_results.append(alg_obj_val)
        r1s.append(r1)
        r2s.append(r2)
        cluster_ct = len(clustering)
        cluster_cts.append(cluster_ct)
        if r1 == r2:
            equal_radii.append(r1)
            obj_vals_equal_radii.append(alg_obj_val)
plt.scatter(equal_radii, obj_vals_equal_radii)
plt.title("Facebook0")
plt.xlabel("radius")
plt.ylabel("objective value")
plt.show()
print('Parameters attaining minimum objective value')
opt = min(best_results)
for k in range(len(best_results)):
    if best_results[k] == opt:
        print('minimum obj val = ',best_results[k], 'num of clusters = ',cluster_cts[k], 'r1 = ',r1s[k], 'r2 = ',r2s[k])


# In[7]:


distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val = exact(fb_adj_mx, 0.7, 0.7)


# In[8]:


clustering, cluster_clock = cluster(distances, L_t_vals, neighborsR, neighborsR2, 0.7, 0.7)


# In[9]:


disag_vector, alg_obj_val, obj_vx = LocalObj(fb_adj_mx, clustering, degrees, math.inf)


# In[10]:


print('frac val = ',frac_val)
print('objective value =', alg_obj_val)
print('time1 = ',clock)
print('time2 = ',cluster_clock)
print('total time =', clock + cluster_clock)


# In[11]:


print(len(clustering))
cluster_szs = []
for i in range(len(clustering)):
    cluster_szs.append(len(clustering[i]))
print(max(cluster_szs))


# In[12]:


print('Maximum Circle Containment for Large Clusters')
for j in range(len(cluster_szs)):
    if cluster_szs[j] >= 10:
        arr = []
        clus = clustering[j]
        clus_orig_names = []
        for i in range(len(clus)):
            clus_orig_names.append(names[clus[i]])
        for k in range(len(circles)):
            count = 0
            for i in range(len(clus_orig_names)):
                if clus_orig_names[i] in circles[k]:
                    count = count + 1
            arr.append(count)
        print('cluster size =', cluster_szs[j], 'max_containment =', max(arr), 'proportion =', max(arr)/cluster_szs[j], 'k = ', np.argmax(arr))


# In[159]:


tot = 0
for i in range(500):
    pivot_clusters, pivot_clusters_list = PivotAlg(fb_adj_mx)
    pivot_disag_vector, pivot_alg_obj_val, pivot_obj_vx = LocalObj(fb_adj_mx, pivot_clusters_list, degrees, math.inf)
    tot = tot + pivot_alg_obj_val
print('avg pivot objective value = ', tot/500)

