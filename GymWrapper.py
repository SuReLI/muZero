import gymnasium as gym
import torch 

class GymWrapper():
    def __init__(self,env_id): #takes as input the name of the desired environment (for example "CartPole-v1")
        self.env=gym.make(env_id)

        
    def initial_state(self) -> torch.Tensor :
        return self.env.reset()[0] #gym reset() method returns obs, info

    
    def step(self,action) -> tuple[float, torch.Tensor, bool]:
        o,r,t,_,_=self.env.step(action)
        return r,o,t
    
    def mask(self) -> torch.Tensor:
        return torch.ones(self.env.single_action_space.shape) #allow all actions for a simple environment
    
