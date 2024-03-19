import numpy as np

from planning import planning

PRINT_LOGS = False
N_SIMULATION = 500

ACTION_SPACE_SIZE = 4
STATE_SPACE_SIZE = 10


# representation model : identity
def h(observations):
    current_state = observations[-1]
    return current_state


# dynamics model : random dynamic
def dynamic(state, action):
    random_reward = np.random.randn()
    random_next_state = np.random.rand(STATE_SPACE_SIZE)
    return random_reward, random_next_state


# prediction model : random prediction
def prediction(state):
    random_policy = np.random.rand(ACTION_SPACE_SIZE)
    random_policy = random_policy / random_policy.sum()
    random_value = np.random.randn()
    return random_policy, random_value


def main():
    # 10 observations
    observations = [np.random.rand(STATE_SPACE_SIZE) for _ in range(10)]


    print("Start MCTS planning ...")
    nu, policy_MCTS = planning(
        h=h,
        dynamic=dynamic,
        prediction=prediction,
        o=observations,
        n_simulation=N_SIMULATION,
        debug=PRINT_LOGS,
    )
    print("Done :)")
    print("Policy MCTS : ", policy_MCTS)


if __name__ == "__main__":
    main()
