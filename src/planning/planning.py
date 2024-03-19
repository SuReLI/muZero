import numpy as np


class Node:
    def __init__(self, P, state_representation=None, reward=None, N=1):
        """
        Represents a Node in the MCTS tree.

        Parameters
        ----------
        P : the probability of going from the parent node to this node. Given by the model policy.
        state_representation : the state representation of this node.
        reward : the reward obtained going from the parent node to this node.
        N : the visit count i.e. the number of times this node has been visited.
        """

        self.P = P
        self.state_representation = state_representation
        self.reward = reward
        self.N = N

        # Initialize the Q value to 0
        self.Q = 0

        # Initialize the children of this node to an empty dictionary
        self.children = {}


def upper_confidence_bound(
    parent: Node, child: Node, c1: float = 1.25, c2: float = 19652
):
    """
    Computes the upper confidence bound of the child node.

    Parameters:
    ----------
    parent : the parent node.
    child : the child node. **The UCB is computed for this node.**
    c1 : an exploration/exploitation tradeoff parameter. Set in the muZero paper to 1.25.
    c2 : an exploration/exploitation tradeoff parameter. Set in the muZero paper to 19652.

    """
    puct = child.Q + child.P * (np.sqrt(parent.N) / (1 + child.N)) * (
        c1 + np.log((parent.N + c2 + 1) / c2)
    )
    return puct


def select_next_node(node: Node):
    """
    Starting from a node, selects the children greedly with respect to the UCB.

    Parameters:
    ----------
    node : the node to select the next child from.

    Returns:
    -------
    The next child, and the action that leads to it.

    """
    _, next_action, next_child = max(
        (upper_confidence_bound(node, child), action, child)
        for action, child in node.children.items()
    )
    return next_child, next_action


def selection_phase(root: Node, debug: bool = False):
    """
    Implements the selection phase of the MCTS algorithm.
    Starting from a root node, goes down the tree selecting nodes greddily with respect to the UCB.

    Parameters:
    ----------
    root : the root node of the tree. Where the selection starts.
    debug : whether to print logs or not.

    Returns:
    -------
    The leaf node, the trajectory of nodes selected, and the history of actions taken.
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


def expansion_phase(
    leaf_node: Node,
    parent: Node,
    action: float,
    dynamic: callable,
    prediction: callable,
    debug: bool = False,
):
    """
    Expands the tree at the leaf node. Leverages the model.

    Parameters:
    ----------
    leaf_node : the node to expand.
    parent : the parent node of the leaf node.
    action : the action that leads from the parent node to the leaf node.
    dynamic : the model dynamic function.
    prediction : the model prediction function.
    debug : whether to print logs or not.

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


def backup_phase(
    trajectory: list[Node], value: float, gamma: float = 0.99, debug: bool = False
):
    """
    Updates the Q values of the edges in the trajectory.

    Parameters:
    ----------
    trajectory : the trajectory of nodes selected.
    value : the value of the leaf node.
    gamma : the discount factor.
    debug : whether to print logs or not.
    """
    global minQ, maxQ

    reversed_traj = trajectory[::-1]
    for k, node in enumerate(reversed_traj):
        if debug:
            print("Backup step : ", k)

        # Compute iteratively the discounted return G
        if k == 0:
            G = value
        else:

            reward_to_add = reversed_traj[k - 1].reward
            G = reward_to_add + gamma * G

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


def planning(
    h: callable,
    dynamic: callable,
    prediction: callable,
    o: list[np.array],
    n_simulation: int = 10,
    debug: bool = False,
):
    """
    The main function implementing the MCTS algorithm.

    Parameters:
    ----------
    h : the model representation function.
    dynamic : the model dynamic function.
    prediction : the model prediction function.
    o : the past observations
    n_simulation : the number of simulations to run.
    debug : whether to print logs or not.
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
