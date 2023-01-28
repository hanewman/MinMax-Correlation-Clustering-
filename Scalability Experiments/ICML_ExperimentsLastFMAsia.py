#!/usr/bin/env python
# coding: utf-8




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





#Read in the data
df = pd.read_csv('LastFMAsia.csv')

sources = list(df["NODE1"])
targets = list(df["NODE2"])





#Construct positive adjacency matrix 
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





degrees = DegreeDist(fb_adj_mx)





distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val = exact(fb_adj_mx, 0.7, 0.7)





clustering, cluster_clock = cluster(distances, L_t_vals, neighborsR, neighborsR2, 0.7, 0.7)





disag_vector, alg_obj_val, obj_vx = LocalObj(fb_adj_mx, clustering, degrees, math.inf)

