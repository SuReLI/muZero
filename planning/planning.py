import numpy as np

class Node():
    def __init__(self, P, state_representation=None, state_value=None, reward=None, N=0):
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
        self.state_value = state_value
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




from model import dynamics, prediction

def expansion_phase(leaf_node, action_chosen, f, g):
    """
    Expands the tree from the leaf node. Leverages the model.
    
    Parameters:
    ----------
    leaf_node : The node to expand.
    action_chosen : The action chosen from the leaf node.
    
    """
    
    # The reward and state are computed by the *dynamics* function
    r, s = dynamics(leaf_node.state_representation, action_chosen)
    # The policy and value are computed by the *prediction* function
    policy, value = prediction(s)
    
    # A new node is selected and added to the search tree
    new_node 
    
    # Initialize the edges stemming from the new node
    
    
    return None

def backup_phase(trajectory, reward):
    """
    Updates the Q values of the edges in the trajectory.
    """
    return None