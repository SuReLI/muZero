import numpy as np

# from model import dynamics, prediction


class Node:
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

        self.children = {}  # dictionnay of action : Node


def upper_confidence_bound(parent, child, c1=1.25, c2=19652):
    """
    Returns the upper confidence bound of the edge.
    """

    return child.Q + child.P * (np.sqrt(parent.N) / (1 + child.N)) * (
        c1 + np.log((parent.N + c2 + 1) / c2)
    )


def select_next_node(node):
    """
    Starting from a node, selects the most promising edge based on the UCB.
    """
    _, next_action, next_child = max(
        (upper_confidence_bound(node, child), action, child)
        for action, child in node.children.items()
    )
    return next_child, next_action


def selection_phase(root, debug=False):
    """
    Implements the selection phase of the MCTS algorithm.
    Starting from a root node, goes down the tree selecting nodes greddily with respect to the UCB.

    Returns the trajectory of nodes, egdes selected.
    """
    current_node = root
    trajectory = [root]
    history = []

    # while the current node is not a leaf node == is a parent node == has children
    if debug:
        print("Going down the tree toward a leaf node ...")
    while current_node.children:
        next_node, next_action = select_next_node(current_node)

        trajectory.append(next_node)
        history.append(next_action)

        current_node = next_node

    leaf_node = current_node

    return leaf_node, trajectory, history


def expansion_phase(leaf_node, parent, action, dynamic, prediction, debug=False):
    """
    Expands the tree at the leaf node. Leverages the model.

    Parameters:
    ----------
    leaf_node : The node to expand.
    parent : The parent node of the leaf node.
    action : The action that leads from the parent node to the leaf node.
    dynamic : The model dynamics function.
    prediction : The model prediction function.

    Returns:
    -------
    The value of the leaf node - predicted by the model prediction function.
    """

    # Use the model *dynamic*
    if debug:
        print("Using the model dynamic ...")
    leaf_node.reward, leaf_node.state_representation = dynamic(
        parent.state_representation, action
    )

    # Use the model *prediction*
    if debug:
        print("Using the model prediction ...")
    policy, value = prediction(leaf_node.state_representation)

    # Lazy expansion of the leaf node
    if debug:
        print("Expanding the leaf node ...")
    for action in range(len(policy)):
        hypothetical_next_node = Node(policy[action])
        leaf_node.children[action] = hypothetical_next_node

    return value


def backup_phase(trajectory, value, gamma=0.99, debug=False):
    """
    Updates the Q values of the edges in the trajectory.

    Parameters:
    ----------
    trajectory :List[Node], The trajectory of nodes selected.
    value : The value of the leaf node.
    gamma : The discount factor.
    """
    global minQ, maxQ

    reversed_traj = trajectory[::-1]
    for k, node in enumerate(reversed_traj[:-1]):
        if debug:
            print("Backup step : ", k)

        # Compute iteratively the discounted return G
        if k == 0:
            G = value
        else:

            G = node.reward + gamma * G

        # Update the Q value of the edge and its visit count
        node.N += 1
        node.Q = ((node.N * node.Q) + G) / (node.N + 1)

        # Check if the Q value is a new max or min for the tree
        if node.Q > maxQ:
            maxQ = node.Q
        if node.Q < minQ:
            minQ = node.Q

        # Normalize the Q values
        node.Q = (node.Q - minQ) / (maxQ - minQ) if maxQ != minQ else node.Q

    return None


def planning(h, dynamic, prediction, o, n_simulation=10, debug=False):
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

    global minQ, maxQ
    minQ = np.inf
    maxQ = -np.inf

    # initializing the tree
    if debug:
        print("Initializing the tree ...")

    policy, value = prediction(root.state_representation)

    for action in range(len(policy)):
        hypothetical_next_node = Node(policy[action])
        root.children[action] = hypothetical_next_node

    for sim in range(n_simulation):
        if debug:
            print("-" * 10 + f"Simulation : {sim+1}" + "-" * 10)

        ########
        if debug:
            print("SELECTION phase")
        leaf_node, trajectory, history = selection_phase(root, debug=debug)
        ########
        if debug:
            print("EXPANSION phase")
        value = expansion_phase(
            leaf_node, trajectory[-1], history[-1], dynamic, prediction, debug=debug
        )
        ########
        if debug:
            print("BACKUP phase")
        backup_phase(trajectory, value, debug=debug)

    # Compute MCTS policy
    if debug:
        print("Computing MCTS policy")
    policy_MCTS = [children.N for children in root.children.values()]
    policy_MCTS = policy_MCTS / np.sum(policy_MCTS)

    # Final state value "nu"
    nu = root.Q
    # nu_2 = np.sum(
    #    [
    #        policy_MCTS[i] * children.Q
    #        for i, children in enumerate(root.children.values())
    #    ]
    # )

    return nu, policy_MCTS
