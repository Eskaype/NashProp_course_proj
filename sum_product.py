import numpy as np
from random import choice
import networkx as nx
from fglib import graphs, nodes, rv,inference
from fglib.utils import draw_message,draw
import matplotlib.pyplot as plt
#from inference import sum_product

def make_debug_graph(num_nodes,graph_mat):
    # Create factor graph
    fg = graphs.FactorGraph()
    # Create variable nodes
    X  = {}
    F=  {}
    dist_fa = np.zeros([30,2,2])
    for k in range(30):
        dist_fa[k] = np.random.rand(2,2)
    for i in range(num_nodes):
        X['x'+str(i+1)] = nodes.VNode("x"+str(i+1),rv.Discrete)


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
    draw(fg,nx.circular_layout(fg))
    plt.show()
    print([attr["type"].value for (n, attr) in fg.nodes(data=True)])
    belief = inference.loopy_belief_propagation(fg,1000,[*X.values()],order = [*F.values(),*X.values()])
    print(X['x1'].belief())
    print(X['x2'].belief())
    print(X['x3'].belief())
    ## Now assign values to the factors ?

#graph_mat = [(1,5),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8),(3,5),(3,6),(3,7),(4,6),(4,7),(4,8)]

def make_debug_graph_new():
    fg = graphs.FactorGraph()
    x1 = nodes.VNode('x1',rv.Discrete)
    x2 = nodes.VNode('x2', rv.Discrete)
    x3 = nodes.VNode('x3', rv.Discrete)
    dist_fa = [[0.1, 0.7],[0.3,0.2]]
    dist_f2 = [0.3,0.7]
    f1 = nodes.FNode('f1', rv.Discrete(dist_f2, x1))
    f2 = nodes.FNode('f2', rv.Discrete(dist_fa, x1,x2))
    #f3 = nodes.FNode('f3', rv.Discrete(dist_fa, x1,x2 ))
    f4 = nodes.FNode('f4', rv.Discrete(dist_fa, x2,x3))
    print(f1.factor)
    fg.set_nodes([x1, x2, x3])
    fg.set_nodes([f1, f2,f4])
    # Add edges to factor graph
    fg.set_edge(f1, x1)
    fg.set_edge(x1, f2)
    fg.set_edge(f2, x2)
    #fg.set_edge(x2, f3)
    #fg.set_edge(f3, x1)
    fg.set_edge(x2, f4)
    fg.set_edge(x3, f4)
    draw(fg,nx.circular_layout(fg))
    plt.show()
    belief = inference.loopy_belief_propagation(fg, 10, [x1,x2,x3],order = [f1,f2,f4,x1,x2,x3])

graph_mat= []
N=4
for i in range(1,30):
    r = i%29
    print(r)
    graph_mat.append((i,i%29+1))
#graph_mat = [(1,2),(2,3),(3,1)]
print(graph_mat)
#belief = make_debug_graph(29,graph_mat)
belief = make_debug_graph_new()
#Now compute the marginal