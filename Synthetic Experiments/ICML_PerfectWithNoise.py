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
import csv

from ipynb.fs.full.MinMaxCorrelationClustering import exact
from ipynb.fs.full.MinMaxCorrelationClustering import cluster
from ipynb.fs.full.MinMaxCorrelationClustering import LocalObj
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLP
from ipynb.fs.full.MinMaxCorrelationClustering import PivotAlg
from ipynb.fs.full.MinMaxCorrelationClustering import DegreeDist
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLPonly
from ipynb.fs.full.MinMaxCorrelationClustering import MinMaxLPneighbors


# In[2]:


#Create a graph with n vertices and n/k cliques of size k 
def PerfectCluster(n,k):
    if n % k != 0:
        raise Exception('k does not divide n')
    adj_mx = np.zeros([n,n])
    num = int(n/k)
    for i in range(num):
        for j in range(k):
            for l in range(k):
                u = int(k*i + j)
                v = int(k*i + l)
                adj_mx[u][v] = 1
    return adj_mx


# In[3]:


#Output the cliques created in the PerfectCluster function as a list of lists
def Circles(n,k):
    if n % k != 0:
        raise Exception('k does not divide n')
    circles = []
    for i in range(k):
        circle = []
        for j in range(k):
            circle.append(k*i + j)
        circles.append(circle)
    return circles           


# In[5]:


#given the positive adjacency matrix, flip random edges to opposite sign
#number of flips = level*(#positive edges) 
#output the adjacency matrix post-flips, and the number of flips 
def Noise(adj_mx, level):
    n = np.shape(adj_mx)[0]
    a = np.dot(adj_mx, np.ones(n))
    degree_sum = np.dot(a, np.ones(n))
    flips = math.floor(level*degree_sum/2)
    for i in range(flips):
        rand1, rand2 = np.random.randint(0, n , size=2)
        if rand1 != rand2:
            if adj_mx[rand1][rand2] == 0:
                adj_mx[rand1][rand2] = 1
                adj_mx[rand2][rand1] = 1
            else:
                adj_mx[rand1][rand2] = 0
                adj_mx[rand2][rand1] = 0 
    return adj_mx, flips    


# In[6]:


step_sz = 0.10
upper_bd = 2.0
n = 100 

degree_vecs = []
cluster_ct_vec_exact = []
cluster_ct_vec_LP = []
cluster_ct_vec_pivot = []
max_degree_vec = []
alg_obj_val_vec_exact = []
alg_obj_val_vec_LP = []
alg_obj_val_vec_pivot = []
frac_val_vec_exact = []
frac_val_vec_LP = []

#Run our exact algorithm and the KMZ algorithm on a perfect clustering exposed to increasing levels of noise
for j in range(int(upper_bd/step_sz)):
    B = PerfectCluster(n, int(math.sqrt(n)))
    C, flips = Noise(B, j*step_sz)
    
    distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val = exact(C, 0.7, 0.7)
    clustering, cluster_clock = cluster(distances, L_t_vals, neighborsR, neighborsR2, 0.7, 0.7)
    degrees = DegreeDist(B)
    degree_vecs.append(degrees)
    disag_vector, alg_obj_val, obj_vx = LocalObj(C, clustering, degrees, math.inf)
    
    cluster_ct  = len(clustering)
    cluster_ct_vec_exact.append(cluster_ct)
    max_degree_vec.append(max(degrees))
    alg_obj_val_vec_exact.append(alg_obj_val)
    frac_val_vec_exact.append(frac_val)

    #Print the clustering output by our exact algorithm
    print('start' + str(j))
    for i in range(cluster_ct):
        print(clustering[i])
    
    
    LP_objVal, LP_distances, LP_L_t_vals, LP_neighborsR, LP_neighborsR2, LP_clock = MinMaxLP(C, 0.4, 0.4, -1)
    LP_clustering, LP_cluster_clock = cluster(LP_distances, LP_L_t_vals, LP_neighborsR, LP_neighborsR2, 0.4, 0.4)
    LP_disag_vector, LP_alg_obj_val, LP_obj_vx = LocalObj(C, LP_clustering, degrees, math.inf)
    
    LP_cluster_ct = len(LP_clustering)
    cluster_ct_vec_LP.append(LP_cluster_ct)
    alg_obj_val_vec_LP.append(LP_alg_obj_val)
    frac_val_vec_LP.append(LP_objVal)
    
    pivot_clusters, pivot_clusters_list = PivotAlg(C)
    pivot_disag_vector, pivot_alg_obj_val, pivot_obj_vx = LocalObj(C, pivot_clusters_list, degrees, math.inf)
    alg_obj_val_vec_pivot.append(pivot_alg_obj_val)
    cluster_ct_vec_pivot.append(len(pivot_clusters))
    


# In[9]:


#Plots for the above experiments
#Uncomment to display the relevant plot

levels = []
for i in range(int(upper_bd/step_sz)):
    levels.append(i*step_sz*450)
fig, axs = plt.subplots(1)

#cluster_cts
# plt.scatter(levels, cluster_ct_vec_exact,  c='b', label = 'cluster counts')
# plt.scatter(levels, cluster_ct_vec_LP , c='r', label = 'KMZ cluster counts')
# plt.scatter(levels, cluster_ct_vec_pivot , c='y', label = 'Pivot cluster counts')


#obj_vals_w_pivot
# plt.scatter(levels, alg_obj_val_vec_exact, c = 'b', label = 'objective value')
# plt.scatter(levels, alg_obj_val_vec_LP, c = 'r', label = 'KMZ objective value')
# plt.scatter(levels, alg_obj_val_vec_pivot, c = 'g', label = 'Pivot objective value')
# plt.scatter(levels, max_degree_vec, c = 'y', label = 'max degree')

#frac_vals
# plt.scatter(levels, frac_val_vec_exact, c = 'b', label = 'fractional cost')
# plt.scatter(levels, frac_val_vec_LP, c = 'r', label = 'LP objective value')

ratios_frac_LP = []
ratios_alg_LP = []
ratios_alg_frac = []
ratios_LPalg_LP = []
for i in range(int(upper_bd/step_sz)):
    if frac_val_vec_exact[i] == 0:
        ratios_frac_LP.append(1)
    else:
        ratios_frac_LP.append(frac_val_vec_exact[i]/frac_val_vec_LP[i])
        
    if alg_obj_val_vec_exact[i] == 0:
        ratios_alg_LP.append(1)
        ratios_alg_frac.append(1)
    else:
        ratios_alg_LP.append(alg_obj_val_vec_exact[i]/frac_val_vec_LP[i])
        ratios_alg_frac.append(alg_obj_val_vec_exact[i]/frac_val_vec_exact[i])
        
    if alg_obj_val_vec_LP[i] == 0:
        ratios_LPalg_LP.append(1)
    else:
        ratios_LPalg_LP.append(alg_obj_val_vec_LP[i]/frac_val_vec_LP[i])

#all
# plt.scatter(levels, alg_obj_val_vec_exact, c = 'b', label = 'objective value')
# plt.scatter(levels, alg_obj_val_vec_LP, c = 'r', label = 'KMZ objective value')
# plt.scatter(levels, frac_val_vec_exact, c = 'g', label = 'fractional cost')
# plt.scatter(levels, frac_val_vec_LP, c = 'y', label = 'LP objective value')
# plt.scatter(levels, max_degree_vec, c = 'orange', label = 'max degree')
# plt.scatter(levels, alg_obj_val_vec_pivot, c = 'violet', label = 'Pivot objective value')

#alg_and_frac_to_LP
# plt.scatter(levels, ratios_frac_LP, c = 'b', label = 'fractional cost / LP objective value')
# plt.scatter(levels, ratios_alg_LP, c = 'r', label = 'objective value / LP objective value')

#rounding_ratios
# plt.scatter(levels, ratios_alg_frac, c = 'b', label = 'objective value / fractional cost')
# plt.scatter(levels, ratios_LPalg_LP, c = 'r', label = 'KMZ objective value / LP objective value')

plt.xlabel('number of flips')
plt.legend()


fig.tight_layout()


# In[10]:


#compute portion of a cluster contained in a ground truth circle 
def proportion(clus):
    circles = Circles(10,10)
    proportions = []
    for i in range(len(circles)):
        count = 0
        for j in range(len(clus)):
            if clus[j] in circles[i]:
                count = count + 1
        proportions.append(count/len(clus))
    return max(proportions)


# In[11]:


#quantify how much of a cluster is contained in a ground truth circle
count = 0
for line in open("synthetic_clusters.csv"):
    line = line.strip()
    csv_row = line.split(',')
    if csv_row[0] == '1000' or count == 0:
        count = count + 1
        arr = ['start' + str(count)]
        #print(arr)
    else:
        new_row = []
        for i in range(len(csv_row)):
            if csv_row[i] != '':
                new_row.append(int(csv_row[i]))
        if len(new_row) >= 1:
            if int(proportion(new_row)) != 1:
                print('no')
                print(proportion(new_row))


# In[ ]:




