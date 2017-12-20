"""Module for inference algorithms.

This module contains Nash Propagation algorithm to perform inference 

Functions:

    Nash_propagation : Nash Propagation ---------------- This is an addition done for course project 
"""

from random import choice
import networkx as nx
from fglib import rv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from . import nodes

def nash_propagation(model,iterations,epsilon,tau,query_node=(),order = None):

"""  Perform initialization of variable to factor nodes 
"""
    for n in order:
        if(n.type.value==1):
            for neighbor in nx.all_neighbors(model,n):
                table = rv.Discrete(np.ones([10,10]),n,neighbor)
                model[n][neighbor]['object'].set_table(n, neighbor, table)
        if(n.type.value == 2):
            for neighbor in nx.all_neighbors(model, n):
                table = rv.Discrete(np.ones([10, 10]),n,neighbor)
                model[n][neighbor]['object'].set_table(n, neighbor, table)
    step =0
    grid={}
"""
   epa : expected payoff  computation 
   order : order for the puprose of the project has been limited to starting with all factor nodes and then updating variable nodes
"""
    for _ in range(iterations):
        print(step)
        step =step+1
        # Visit nodes in predefined order
        for n in order:
            #print(n)
            for neighbor in nx.all_neighbors(model, n):
                 table = n.epa(neighbor,tau,epsilon)  ## Get message from the neighbour and update
                 #Set Messgae
                 #if(n.type.value==2):
                 #   print("")
                 if(n.type.value==1):
                     grid[str(n)] = table.pmf
                 model[n][neighbor]['object'].set_table(n, neighbor, table)
            #if the node is a variable node perform a cross product

            # If the node is a factor node just set the message it passes to its neighbour
        #print(grid)
        sub=0
        #### Uncomment the below codes for plotting 
        plt.imshow(grid['x1'], cmap='gray', vmin=0, vmax=1)
        plt.show()
        print(grid)
        for i in grid.keys():
            sub=sub+1
            plt.subplot(len(grid.keys()),1,sub)
            plt.imshow(grid[i],cmap='gray', vmin = 0, vmax = 1)
        plt.show()
        sub =0
        for i in grid.keys():
            sub=sub+1
            plt.subplot(len(grid.keys()),1,sub)
            plt.imshow(np.ones(grid[i].shape),cmap='gray', vmin = 0, vmax = 1)
        plt.show()
    return grid


