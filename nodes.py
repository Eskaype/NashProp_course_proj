"""Module for nodes of factor graphs.

This module contains classes for nodes of factor graphs,
which are used to build factor graphs.

Classes:
    Node: Abstract class for nodes.
    VNode: Class for variable nodes.
    IOVNode: Class for custom input-output variable nodes.
    FNode: Class for factor nodes.
    IOFNode: Class for custom input-output factor nodes.

"""

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from types import MethodType

import networkx as nx
import numpy as np

from . import rv


class NodeType(Enum):

    """Enumeration for node types."""

    variable_node = 1
    factor_node = 2


class Node(metaclass=ABCMeta):

    """Abstract base class for all nodes."""

    def __init__(self, label):
        """Create a node with an associated label."""
        self.__label = str(label)
        self.__graph = None

    def __str__(self):
        """Return string representation."""
        return self.__label

    @abstractproperty
    def type(self):
        pass

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        self.__graph = graph

    def neighbors(self, exclusion=None):
        """Get all neighbors with a given exclusion.

        Return iterator over all neighboring nodes
        without the given exclusion node.

        Positional arguments:
        exclusion -- the exclusion node

        """

        if exclusion is None:
            return nx.all_neighbors(self.graph, self)
        else:
            # Build iterator set
            iterator = (exclusion,) \
                if not isinstance(exclusion, list) else exclusion

            # Return neighbors excluding iterator set
            return (n for n in nx.all_neighbors(self.graph, self)
                    if n not in iterator)

    @abstractmethod
    def spa(self, tnode):
        pass

    @abstractmethod
    def mpa(self, tnode):
        pass

    @abstractmethod
    def msa(self, tnode):
        pass

    @abstractmethod
    def mf(self, tnode):
        pass
    @abstractmethod
    def epa(self,tnode):
        pass

class VNode(Node):

    """Variable node.

    Variable node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, init=None,projection=None, observed=False):
        """Create a variable node."""
        super().__init__(label)
        if init is not None:
            self.init = init
        else:
            self.init = rv.Discrete(np.array([1, 1]), self)
        self.observed = observed
        self.__projection = projection

    @property
    def type(self):
        return NodeType.variable_node

    @property
    def init(self):
        return self.__init

    @init.setter
    def init(self, init):
        self.__init = init
        
    def compute_proj(self,tnode):
        #iterate over Tables for all its neighbours
        T = self.__projection
        for n in self.neighbors(tnode):
            T= np.outer(self[tnode][n]['object'].get_table(tnode, n),T)
        return
    def set_table(self,table):
        self.__projection = table
        return
    def get_projection(self):
        return  self.__projection
    def belief(self, normalize=True):
        """Return belief of the variable node.

        Args:
            normalize: Boolean flag if belief should be normalized.

        """
        iterator = self.graph.neighbors(self)

        # Pick first node
        n = next(iterator)

        # Product over all incoming messages
        belief = self.graph[n][self]['object'].get_message(n, self)
        for n in iterator:
            belief *= self.graph[n][self]['object'].get_message(n, self)

        if normalize:
            belief = belief.normalize()

        return belief

    def maximum(self, normalize=True):
        """Return the maximum probability of the variable node.

        Args:
            normalize: Boolean flag if belief should be normalized.

        """
        b = self.belief(normalize)
        return np.amax(b.pmf)

    def argmax(self):
        """Return the argument for maximum probability of the variable node."""
        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        b = self.belief()
        return b.argmax(self)

    def spa(self, tnode):
        """Return message of the sum-product algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init

            # Product over incoming messages
            for n in self.neighbors(tnode):
                msg *= self.graph[n][self]['object'].get_message(n, self)

            return msg

    def mpa(self, tnode):
        """Return message of the max-product algorithm."""
        return self.spa(tnode)

    def msa(self, tnode):
        """Return message of the max-sum algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init
            msg = msg.log()

            # Sum over incoming messages
            for n in self.neighbors(self.graph, self, tnode):
                msg += self.graph[n][self]['object'].get_message(n, self)

            return msg

    def mf(self, tnode):
        """Return message of the mean-field algorithm."""
        if self.observed:
            return self.init
        else:
            return self.belief(self.graph)
    ###### Expected PayOff Algorithm similar to spa or mf 
    """ 
           To compute the message/table passed from the variable to Factor . 
           Outer product of all incoming tables is computed. and table/msg is updated 
    """
    def epa(self,tnode,tau,epsilon):

        if self.observed:
            return self.init
        else:
            # Initial message
            #msg = self.init
            msg = self.get_projection()
            # Product over incoming messages
            for n in self.neighbors(tnode):
                ## Here compute the expected payoff check if the payoffs are smaller than epsilon and update the table+
                ## Take the table passed by each factor node and compute the outer product
                msg_neigh = self.graph[n][self]['object'].get_table(n, self)
                msg = msg*msg_neigh.pmf
            ret = rv.Discrete(msg,self,tnode)
            self.__projection = ret.pmf
            #print(self.get_projection())
            return ret
class IOVNode(VNode):

    """Input-output variable node.

    Input-output variable node inherited from variable node class.
    Overwrites all message passing methods of the base class
    with a given callback function.

    """

    def __init__(self, label, init=None, observed=False, callback=None):
        """Create an input-output variable node."""
        super().__init__(label, init, observed)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = MethodType(callback, self)
        self.mpa = MethodType(callback, self)
        self.msa = MethodType(callback, self)
        self.mf = MethodType(callback, self)


class FNode(Node):

    """Factor node.

    Factor node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, factor=None):
        """Create a factor node."""
        super().__init__(label)
        self.factor = factor
        self.record = {}

    @property
    def type(self):
        return NodeType.factor_node

    @property
    def factor(self):
        return self.__factor

    @factor.setter
    def factor(self, factor):
        self.__factor = factor

    def spa(self, tnode):
        """Return message of the sum-product algorithm."""
        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(tnode):
            msg *= self.graph[n][self]['object'].get_message(n, self)

        # Integration/Summation over incoming variables
        for n in self.neighbors(tnode):
            msg = msg.marginalize(n, normalize=False)

        return msg

    def mpa(self, tnode):
        """Return message of the max-product algorithm."""
        self.record[tnode] = {}

        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(tnode):
            msg *= self.graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.maximize(n, normalize=False)

        return msg

    def msa(self, tnode):
        """Return message of the max-sum algorithm."""
        self.record[tnode] = {}

        # Initialize with logarithm of local factor
        msg = self.factor.log()

        # Sum over incoming messages
        for n in self.neighbors(self.graph, self, tnode):
            msg += self.graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(self.graph, self, tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.max(n)

        return msg

    def mf(self, tnode):
        """Return message of the mean-field algorithm."""
        # Initialize with local factor
        msg = self.factor

#TODO:    # Product over incoming messages
#         for n in self.neighbors(graph, self, tnode):
#             msg *= graph[n][self]['object'].get_message(n, self)
#
#         # Integration/Summation over incoming variables
#         for n in self.neighbors(graph, self, tnode):
#             msg = msg.int(n)

        return msg
    """
     Expected Payoff from variable to Factor Node
        
    """"
    def epa(self, tnode,tau,epsilon):


            # Initial message
        msg = self.factor

            # Product over incoming messages
        for n in self.neighbors(tnode):
                ## Here compute the expected payoff check if the payoffs are smaller than epsilon and update the table+
                ## Get probabilities for which the value in the table =1
            tab_a=[]
            tab_b = []
            probability_a=[]
            T = self.graph[n][self]['object'].get_table(n, self)
            for i in range(len(T.pmf)):
                 if np.any(T.pmf[i]==1):
                    tab_a.append(i)
                 if np.any(np.transpose(T.pmf[i])==1):
                    tab_b.append(i)

            probability_a = [i*tau for  i in tab_a]
            probability_b = [i*tau for i in tab_b]
            index_a = []
            index_b =[]
            index = 0
            Table = self.graph[n][self]['object'].get_table(n,self)
            for p in probability_b:
                 Expected_payoff= np.matmul(np.asarray(msg.pmf),np.array([[p],[1-p]]))
                 if(abs(Expected_payoff[1]-Expected_payoff[0])>epsilon):
                         index_b.append(index)
                 index = index + 1
            index  = 0
            for p in probability_a:
                 Expected_payoff= np.matmul(np.transpose(msg.pmf) ,np.array([[p],[1-p]]))
                 if(abs(Expected_payoff[1]-Expected_payoff[0])>epsilon):
                         index_a.append(index)
                 index = index + 1
            ## Now set the '1's to '0's
            Table = self.graph[n][self]['object'].get_table(n,self)
            for i in index_a:
                for j in index_b:
                    Table.pmf[i,j] = 0


            return Table
        return self.graph[tnode][self]['object'].get_table(tnode,self)


class IOFNode(FNode):

    """Input-output factor node.

    Input-output factor node inherited from factor node class.
    Overwrites all message passing methods of the base class
    with a given callback function.

    """

    def __init__(self, label, factor, callback=None):
        """Create an input-output factor node."""
        super().__init__(self, label, factor)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = MethodType(callback, self)
        self.mpa = MethodType(callback, self)
        self.msa = MethodType(callback, self)
        self.mf = MethodType(callback, self)
