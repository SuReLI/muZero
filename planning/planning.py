class Node():
    def __init__(self, state_representation, state_value, edges):
        self.state_representation = state_representation
        self.state_value = state_value
        
        self.edges = edges

class Edge():
    def __init__(
        self, 
        parent_state,
        P,
        next_state=None, 
        reward=None, 
        N=0,
    ):
        
        """
        Represents an edge in the MCTS tree.

        Parameters
        ----------
        parent_state : Current state from which the edge stems from.
        P : The probability of selecting this edge from the parent state. Given by the model policy.
        next_state : The state that this edge leads to.
        Reward : The reward obtained by taking this edge.
        N : The number of times this edge has been visited.
        """
        
        # tree structure
        self.parent_state = parent_state
        self.P = P
        self.next_state = next_state
        
        
        self.reward = reward
        self.N = N
        self.Q = 0
        
        self.ucb = self.upper_confidence_bound()
        
        
    def upper_confidence_bound(self, c1=1.25, c2=19652):
        """
        Returns the upper confidence bound of the edge.
        """
        
        return 0
    
    
    
    
    
def select_next_node(node):
    """
    Starting from a node, selects the most promising edge based on the UCB.
    """
    
    
    
    return None

def selection_phase(root):
    """
    Implements the selection phase of the MCTS algorithm.
    Starting from a root node, goes down the tree selecting nodes greddily with respect to the UCB.
    
    Returns the trajectory of (node, egde) visited. 
    """
    return None


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
    new_node = 
    
    # Initialize the edges stemming from the new node
    
    
    return None

def backup_phase(trajectory, reward):
    """
    Updates the Q values of the edges in the trajectory.
    """
    return None