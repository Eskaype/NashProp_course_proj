
# coding: utf-8

# In[3]:


import numpy as np
import itertools


# In[2]:


n=5
#Strategy 0 is cooperating,1 is defecting to the left, 2 is defecting to the right
strategy=[]
payoff=[]


# In[36]:


def generate_strategy_permutations(n=3):
    s_all=[]
    for _ in range(n):
        s_all.extend([0,1,2])
    s=itertools.permutations(s_all,n)
    strategy=[]
    for s_ in s:
        v=list(s_)
        if v not in strategy:
            strategy.append(v)
    return strategy


# In[67]:


def generate_payoffs(strategy,n):
    coordinates=[]
    payoffs=[0 for _ in range(n)]
    for i,s in enumerate(strategy):
        if s==2:
            s=-1
        coordinates.append((2*i-s)%(2*n))
    #print(coordinates)
    end=-1
    first_start=-1
    for c in range(2*n):    
        if c in coordinates:
            players=[]
            start=c
            if first_start==-1:
                first_start=start
            #print("start is",start)
            players.extend([i for i,val in enumerate(coordinates) if val==start])
            #print("the players are",players)
            for j in range(c+1,2*n,1):
                if j in coordinates:
                    end=j
                    #print("end is",end)
                    players.extend([i for i,val in enumerate(coordinates) if val==end])
                    #print("the players are",players)
                    break
            total_payoff=end-start
            #print("total payoff is",total_payoff)
            payoff=(total_payoff/float(len(players)))
            #print("individual payoff is",payoff)
            for p in players:
                payoffs[p]+=payoff
    #print("last start is",end)
    players=[]
    players.extend([i for i,val in enumerate(coordinates) if val==first_start])
    players.extend([i for i,val in enumerate(coordinates) if val==end])
    #print("the players are",players)
    total_payoff=(first_start+2*n)-end
    #print("total payoff is",total_payoff)
    payoff=(total_payoff/float(len(players)))
    #print("individual payoff is",payoff)
    for p in players:
        payoffs[p]+=payoff
    #print("all payoffs",payoffs)
    return payoffs
generate_payoffs([0,0,2],3)


# In[45]:




