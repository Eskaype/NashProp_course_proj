import numpy as np
## Build parameters for NashProp
from random import choice
import matplotlib.pyplot as plt
import networkx as nx
from fglib import rv,graphs,inference
from fglib import nodes
from fglib.utils import draw_message,draw


def make_graph():
    G = graphs.FactorGraph()
    x1 = nodes.VNode('x1',rv.Discrete,P)
    x2 = nodes.VNode('x2',rv.Discrete,P)
    x3 = nodes.VNode('x3',rv.Discrete,P)
    x4 = nodes.VNode('x4', rv.Discrete, P)
    x5 = nodes.VNode('x5', rv.Discrete, P)
    dist_fa = np.array([[20,10],[20,80]])
    dist_fb = np.array([[40, 30], [30, 50]])
    dist_fc =  np.array([[30, 50], [40, 30]])
    dist_fd = np.array([[70,10],[10,20]])
    dist_fe = np.array([[120, 50], [80, 100]])
    dist_ff = np.array([[40,100],[70,30]])
    dist_fg = np.array([[20, 10], [50, 30]])
    fa = nodes.FNode('f1', rv.Discrete(dist_fa, x1,x2))
    fb = nodes.FNode('f2', rv.Discrete(dist_fb, x2,x3))
    fc = nodes.FNode('f3', rv.Discrete(dist_fc, x3,x4 ))
    fd = nodes.FNode('f4', rv.Discrete(dist_fd, x4,x1))
    fe = nodes.FNode('f5', rv.Discrete(dist_fe, x4,x5))
    ff = nodes.FNode('f6', rv.Discrete(dist_ff, x5,x3))
    fg = nodes.FNode('f7', rv.Discrete(dist_fg, x1, x3))

    G.set_nodes([x1, x2, x3,x4,x5])
    G.set_nodes([fa,fb, fc,fd,fe,ff,fg])
    # Add edges to factor graph
    G.set_edge(x1, fa)
    G.set_edge(fa, x2)
    G.set_edge(x2, fb)
    G.set_edge(fb, x3)
    G.set_edge(x3, fc)
    G.set_edge(fc, x4)
    G.set_edge(x4, fd)
    G.set_edge(fd, x1)
    G.set_edge(fe, x4)
    G.set_edge(x5, fe)
    G.set_edge(ff,x5)
    G.set_edge(x3, ff)
    G.set_edge(fg,x1)
    G.set_edge(x3, fg)
    draw(G,nx.spectral_layout(G))
    plt.show()
    inference.nash_propagation(G,20,epsilon,1/30,[x1,x2,x3,x4,x5],order = [fa,fb,fc,fd,fe,ff,fg,x1,x2,x3,x4,x5])

def make_debug_graph(num_nodes,graph_mat):
    # Create factor graph
    fg = graphs.FactorGraph()
    # Create variable nodes
    X  = {}
    F=  {}
    dist_fa = np.zeros([len(graph_mat),2,2])
    for k in range(len(graph_mat)):
        dist_fa[k] = np.random.rand(2,2)
    for i in range(num_nodes):
        X['x'+str(i+1)] = nodes.VNode("x"+str(i+1),rv.Discrete,P)


    fg.set_nodes(X.values())
    for i in range(len(graph_mat)):
        edge = graph_mat[i]
        nodez = [X["x" + str(edge[m])] for m in range(len(edge))]
        F['f'+str(i+1)] = nodes.FNode("f"+str(i+1),rv.Discrete(dist_fa[i],*nodez))
    fg.set_nodes(F.values())


    for i in range(len(graph_mat)):
        node_name = graph_mat[i]
        fg.set_edge(  X['x'+str(node_name[0])], F['f'+str(i+1)])
        fg.set_edge( F['f' + str(i+1)],X['x' + str(node_name[1])])

    print([attr["type"].value for (n, attr) in fg.nodes(data=True)])
    draw(fg,nx.spectral_layout(fg))
    plt.show()
    print([*F.values(), *X.values()])
    #belief = inference.loopy_belief_propagation(fg, 1000, [*X.values()], order=[*F.values(), *X.values()])
    grid = inference.nash_propagation(fg, 10, epsilon, 1 / 10, [*X.values()],
                               order=[*F.values(), *X.values()])
    ## COmpute the mixed strategy
    prob =[]
    prob_i=0
    print(len(grid.keys()))
    length =0
    for i in grid.keys():
        if(np.where(grid[str(i)]==1)!=[]):
            length = length+1

        prob_i = np.where(grid[str(i)]==1)
        print('new')
        print(prob_i)

    ## Now assign values to the factors ?

Num_Of_Players = 50
max_degree = 5
epsilon = 0.26 ## To be tuned later
tau = epsilon/(np.power(2,(max_degree+1)*max_degree*np.log(max_degree)))
# Initialize tables for each Node in the graph
fg = graphs.FactorGraph()
P = np.ones([10,10])

Mat_link=[]
for i in range(1,9):
    Mat_link.append((i, i % 9 + 1))
    Mat_link.append((i, np.random.randint(i+1,10)))

print(Mat_link)
#make_debug_graph(10,graph_mat=Mat_link)
make_graph()