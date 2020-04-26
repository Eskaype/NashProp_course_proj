## NashProp_Loopy_Belief_Propagation

This repository implements the `Loopy belief propagation to compute approximate nash` in *pytorch* framework based on the paper "*[Nash Propagation for Loopy Graphical Games](https://www.cis.upenn.edu/~mkearns/papers/nashprop.pdf)"


Files Included 
1. Create_game.ipynb : This is based on Gambit library. We precompute for Hotellings Game the Nash Equilibria using gambits internal polynomial solver. 

2. NashProp.py : Create any arbit graphs using fglib; initialize with payoff and other algorithm paramters  ; Pass the graph structure to the inference algorithm 
    ---- Visualization done using NetworkX library


3. Inference.py : Performs the loops until convergence. Iterates through the nodes in the order provided by the code. Runs through factors and then Variable nodes to update the table 

4. Node.py: EPA algorithm is added into the library . EPA or the expected payoff algorithm. The Msg passing mechanism differs slightly between factor and Variable Nodes

5. Edge.py: The edge tables have been added into the class. These tables are analgous to message passing in probabilistic inference models


![NASH POSTER](https://github.com/Eskaype/NashProp_course_proj/blob/master/nash-prop.png)

