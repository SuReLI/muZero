import gymnasium as gym
import torch 
from gymnasium import spaces 
import numpy as np 
import random

class GymWrapper():
    def __init__(self,env_id): #takes as input the name of the desired environment (for example "CartPole-v1")
        self.env=gym.make(env_id)
        self.observation_space_size = self.env.observation_space.shape[0]
        self.number_actions = self.env.action_space.n

    def initial_state(self) -> torch.Tensor :
        return self.env.reset()[0] #gym reset() method returns obs, info

    
    def step(self,action) -> tuple[float, torch.Tensor, bool]:
        o,r,t,_,_=self.env.step(action)
        return r,o,t
    
    def mask(self) -> torch.Tensor:
        return torch.ones(self.env.action_space.shape) #allow all actions for a simple environment

class MazeEnv(gym.Env):
    def __init__(self, width=10, height=10):
        super(MazeEnv, self).__init__()
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # Actions: 0=up, 1=down, 2=left, 3=right
        self.exit=(8,9)
        self.maze = self.generate_maze()


    def generate_maze(self):
        maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        stack = [(0, 0)]
        maze[0][0] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            
            if x > 1 and maze[y][x - 2]:
                neighbors.append((x - 2, y))
            if x < self.width - 2 and maze[y][x + 2]:
                neighbors.append((x + 2, y))
            if y > 1 and maze[y - 2][x]:
                neighbors.append((x, y - 2))
            if y < self.height - 2 and maze[y + 2][x]:
                neighbors.append((x, y + 2))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[ny][nx] = 0
                maze[(ny + y) // 2][(nx + x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def reset(self):
        self.current_pos = (0, 0)
        return np.array(self.maze)

    def step(self, action_1hot):
        action=np.argmax(action_1hot)
        action=action_1hot
        x, y = self.current_pos

        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.height - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.width - 1:
            x += 1
        print(f'next case {self.maze[x][y]}')
        if self.maze[y][x] == 1:  # Wall
            reward = -1
        elif (x,y) == self.exit:  # Reached the goal
            reward = 1
        else:
            reward = 0
            self.current_pos = (x,y)
            
        done = (x,y) == self.exit
        print(self.current_pos)
        return  reward, np.array(self.maze), done

    def render(self, mode='human'):
        maze_with_agent = [row[:] for row in self.maze]
        x, y = self.current_pos
        a,b=self.exit
        maze_with_agent[y][x] = "A"  # "A" represents the agent
        maze_with_agent[b][a]="E"
        for row in maze_with_agent:
            print(" ".join("X" if cell == 1 else " " if cell == 0 else "E" if cell == "E" else "A" for cell in row))
        print()
    
    def mask(self) -> torch.Tensor:
        return torch.ones(4)

# # Example usage
# env = MazeEnv(width=10, height=10)
# observation = env.reset()
# env.render()

# for _ in range(20):
#     action = env.action_space.sample()  # Random action
#     print(action)
#     observation, reward, done, info = env.step(action)
#     # print(np.array(env.maze))
#     env.render()
#     if done:
#         break
