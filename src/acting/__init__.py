import random
from planning.planning import planning
import numpy as np


def acting(env, h, g, f, n_simulation=10):
    """
    Inputs
        h, g, f the five networks
        env game environment, class with the following methods:
            initial_state() -> state where
            state: Tensor of size
            step(action) -> reward, state, is_terminal where
            reward: float
            state: Tensor of size
            is_terminal: bool

    return
        Tensor observation, of size
        Tensor policy, of size
        int action
        float reward
        float value
    """
    search_statistics = []
    obs = np.zeros((10, env.observation_space_size), dtype=np.float32)

    o_prev = env.initial_state()  # previous observation
    obs[-1] = o_prev

    is_terminal = False

    while not is_terminal:
        nu_t, pi_t = planning(
            h, g, f, obs, n_simulation
        )  # running MCTS algorithm from the current state o_t
        a = random.choices(range(len(pi_t)), weights=pi_t, k=1)[0]  # Choosing the action randomly according to policy pi_t distribution
        r, o_new, is_terminal = env.step(a)
        one_hot_action = np.zeros(len(pi_t))
        one_hot_action[a] = 1
        search_statistics.append((o_prev, pi_t, one_hot_action, r, nu_t))
        o_prev = o_new
        obs[0:-1] = obs[1:]
        obs[-1] = o_new

    return search_statistics
