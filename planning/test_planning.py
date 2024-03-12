import numpy as np

from planning import planning

action_space_size = 2
state_space_size = 10

# planning(h, g, f, o, n_simulation=10):


# representation model : identity
def h(observations):
    current_state = observations[-1]
    return current_state


# dynamics model : random dynamic
def dynamic(state, action):
    random_reward = np.random.randn()
    random_next_state = np.random.rand(state_space_size)
    return random_reward, random_next_state


# prediction model : random prediction
def prediction(state):
    random_policy = np.random.rand(action_space_size)
    random_policy = random_policy / random_policy.sum()
    random_value = np.random.randn()
    return random_policy, random_value


# 10 observations
observations = [np.random.rand(state_space_size) for _ in range(10)]


print("Start planning ...")
nu, policy_MCTS = planning(
    h, dynamic, prediction, observations, n_simulation=10, debug=True
)
print("easy")
