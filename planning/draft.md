# Search - implementing MCTS

"Similar to AlphaZero, the search is divided into **three stages**, repeated for **a number of simulations**."


**0 - Initialization**

- "Every node of the search tree is associated with an internal state s"
- "For each action $a$ from $s$ there is an edge $(s, a)$ that stores a set of statistics" --> $N, Q, R, S$

- We keep tables for this statictics as well as a "state transition table" and "reward table".


**1 - Selection**
For each simulation :

- start at the **root** $s^0$
- At **hypothetical time-step** $k = 1 \dots l$,we are in state $s^k$. Select an greedy action $a^k$ with respect to the **upper confidence bound** computed with the **stored statistics**.
- Stop when you reach a **leaf node**. This node will next be "expended"

During this "selection" process - selecting a leaf node to expand - we visit 
$l$ state-action pairs.

**2 - Expansion**

USe the dynamics function to 


**3 - Backup**