import numpy as np
from model import dynamics, prediction

class Node():
    def __init__(self, P, state_representation=None, reward=None, N=0):
        """
            Represents a Node in the MCTS tree.

            Parameters
            ----------
            parent_state : Current state from which the edge stems from.
            P : The probability of selecting this edge from the parent state. Given by the model policy.
            next_state : The state that this edge leads to.
            Reward : The reward obtained by taking this edge.
            N : The number of times this edge has been visited.
        """
        
        self.P = P
        self.state_representation = state_representation
        self.reward = reward
        self.N = N
        self.Q = 0
        
        self.children = {} # dictionnay of action : Node


        
def upper_confidence_bound(parent, child, c1=1.25, c2=19652):
    """
    Returns the upper confidence bound of the edge.
    """
    
    return child.Q + child.P * (np.sqrt(parent.N) / (1 + child.N))*(c1 + np.log((parent.N + c2 + 1) / c2))
    
    
def select_next_node(node):
    """
    Starting from a node, selects the most promising edge based on the UCB.
    """
    _,next_action,next_child = max((upper_confidence_bound(node, child), action, child) for action, child in node.children.items())
    return next_child, next_action


def selection_phase(root):
    """
    Implements the selection phase of the MCTS algorithm.
    Starting from a root node, goes down the tree selecting nodes greddily with respect to the UCB.
    
    Returns the trajectory of nodes, egdes selected.
    """
    current_node = root
    trajectory = [root]
    history = []

    while not current_node.children : # while the current node is not a leaf node
        next_node, next_action = select_next_node(current_node)
        trajectory.append(next_node)
        history.append(next_action)
        current_node = next_node    
    
    leaf_node = current_node

    return leaf_node, trajectory, history




def expansion_phase(leaf_node,parent,action,dynamic, prediction):
    """
    Expands the tree from the leaf node. Leverages the model.
    
    Parameters:
    ----------
    leaf_node : The node to expand.
    parent : The parent node of the leaf node.
    action : The action that leads from the parent node to the leaf node.
    dynamic : The model dynamics function.
    prediction : The model prediction function.

    
    """
    leaf_node.reward , leaf_node.state_representation = dynamic(parent.state_representation,action)
    
    policy, value = prediction(leaf_node.state_representation) # The policy and value are computed by the *prediction* function at leaf_node

    for action in range (len(policy)):
        hypothetical_next_node = Node(policy[action])
        leaf_node.children[action] = hypothetical_next_node
    
    return value

def backup_phase(trajectory, value, gamma=0.99):
    """
    Updates the Q values of the edges in the trajectory.

    Parameters:
    ----------
    trajectory :List[Node], The trajectory of nodes selected. 
    value : float, The value of the leaf node.
    """
    global minQ, maxQ

    for k, node in enumerate(reversed(trajectory)):
        if k==0:
            G = value
        else:

            G = node.reward + gamma * G
        
        node.N += 1
        node.Q = ((node.N * node.Q) + G) / (node.N + 1)
        
        if node.Q > maxQ:
            maxQ = node.Q
        if node.Q < minQ :
            minQ = node.Q 
        node.Q = (node.Q - minQ) / (maxQ-minQ) # Normalizing the Q values

    return None


def planning(h,g,f, o, n_simulation = 10):
    """
        The main function implementing the MCTS algorithm.

        Parameters:
        ----------
        h : The model representation function.
        g : The model dynamics function.
        f : The model prediction function.
        o : list of observations
        n_simulation : int, The number of simulations to run.
    """
    T = 1
    root = h(o)
    root = Node(1, root)

    #initializing the tree
    policy, value = prediction(root.state_representation) 

    for action in range (len(policy)):
        hypothetical_next_node = Node(policy[action])
        root.children[action] = hypothetical_next_node

    for sim in range(n_simulation):
        leaf_node, trajectory, history = selection_phase(root)
        value = expansion_phase(leaf_node,trajectory[-2],history[-1],g,f)
        backup_phase(trajectory, value)
    
    # compute MCTS policy
    policy_MCTS = [children.N  for children in root.children.values()]
    policy_MCTS = policy_MCTS / np.sum(policy_MCTS)

    # A choisir comme mu value
    mu = np.sum([ policy_MCTS[i]* children.Q for i,children in enumerate(root.children.values())])
    mu_2 = root.Q

    return mu, policy_MCTS