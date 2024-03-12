import random


def acting(env, h, g_r, g_s, f_v, f_p):
    """
    Inputs
        h, g_r, g_s, f_v, f_p the five networks
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
    observations = []
    policies = []
    actions = []
    rewards = []
    values = []
    
    o_t = env.initial_state()
    observations.append(o_t)
    
    mask = env.mask()
    is_terminal = False
    
    while not is_terminal:
        nu_t, pi_t = planning(h, g_r, g_s, f_v, f_p, o_t, mask) # running MCTS algorithm from the current state o_t
        a = random.choices(range(len(pi_t)), weights = pi_t, k=1)[0] # Choosing the action randomly according to policy pi_t distribution
        r,o,is_terminal = env.step(a)
        observations.append(o)
        policies.append(pi_t)
        actions.append(a)
        rewards.append(r)
        values.append(nu_t)
        
    return observations, policies, actions, rewards, values