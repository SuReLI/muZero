from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


### ENVIRONMENT ###
class AbstractEnvironment(ABC):
    @abstractmethod
    def initial_state() -> Tensor:
        pass

    @abstractmethod
    def step(action) -> tuple[float, Tensor, bool]:
        pass

    @abstractmethod
    def apply_policy(policy: Tensor) -> Tensor:
        pass

    @abstractmethod
    def mask() -> Tensor:
        pass


class DummyEnvironment(AbstractEnvironment):
    def __init__(self):
        self.__state = torch.zeros((11))
        self.__state[5] = 1

    def initial_state(self) -> Tensor:
        state = torch.zeros((11))
        state[5] = 1

        return state

    def apply_policy(policy: Tensor):
        action_index = np.random.choice(policy.size(0), p=policy.numpy())

        action = torch.zeros((2,))
        action[action_index] = 1

        return action

    def step(self, action) -> tuple[float, Tensor, bool]:
        if not action.size() == (2,):
            raise ValueError("Invalid action shape")
        if not action.sum() == 1:
            raise ValueError("Invalid action")

        current_position = torch.argmax(self.__state).item()
        if action[0] == 1:
            # Go left
            current_position -= 1
        if action[1] == 1:
            # Go right
            current_position += 1

        current_position = min(10, max(0, current_position))

        self.__state = torch.zeros((11))
        self.__state[current_position] = 1

        # Return values
        reward = 1 if current_position == 10 else 0
        terminal = current_position == 10

        return reward, self.__state, terminal

    def mask(self) -> Tensor:
        return torch.tensor([1, 1])
