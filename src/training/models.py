from typing import Optional

import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """
    Class for Dynamics model: (s, r) ~ g(s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        layer_count: Optional[int] = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_count = layer_count

        self.input_layer = nn.Linear(state_dim + action_dim, state_dim + action_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(state_dim + action_dim, state_dim + action_dim)
                for _ in range(layer_count)
            ]
        )
        self.output_layer1 = nn.Linear(state_dim + action_dim, state_dim)
        self.output_layer2 = nn.Linear(state_dim + action_dim, 1)

    def forward(self, s, a):
        y = torch.cat([s.to(torch.float), a.to(torch.float)], dim=-1)
        y = torch.relu(self.input_layer(y))
        for layer in self.hidden_layers:
            y = torch.relu(layer(y))
            # maybe add batch norm here
        s_k = self.output_layer1(y)
        r_k = self.output_layer2(y)
        return (r_k, s_k)


class Prediction(nn.Module):
    """
    Class for Prediction model: (p, v) ~ f(s)
    """

    def __init__(
        self,
        state_dim: int,
        policy_dim: Optional[int] = 1,
        value_dim: Optional[int] = 1,
        layer_count: Optional[int] = 4,
    ):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim
        self.policy_dim = policy_dim
        self.value_dim = value_dim

        input_dim = self.state_dim
        hidden_dims = self.state_dim  # TODO: Experiment with different networks
        output_dim = self.policy_dim + self.value_dim

        self.input_layer = nn.Linear(input_dim, hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims, hidden_dims) for _ in range(layer_count)]
        )
        self.output_layer1 = nn.Linear(hidden_dims, self.policy_dim)
        self.output_layer2 = nn.Linear(hidden_dims, 1)

    def forward(self, s_k):
        x = torch.relu(self.input_layer(s_k))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            # maybe add batch norm here
        policy_k = self.output_layer1(x)
        value_k = self.output_layer2(x)
        return (policy_k, value_k)


class Representation(nn.Module):
    """
    Class for Representation model: s_0 = h(o)
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        N: int,
        layer_count: Optional[int] = 4,
    ):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.N = N

        input_dim = self.observation_dim * self.N
        hidden_dims = self.state_dim
        output_dim = self.state_dim

        self.input_layer = nn.Linear(input_dim, hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims, hidden_dims) for _ in range(layer_count)]
        )
        self.output_layer = nn.Linear(hidden_dims, output_dim)

    def forward(self, o):
        o = o.view(o.shape[0],-1)
        x = torch.relu(self.input_layer(o))
        for layer in self.hidden_layers:
            y = torch.relu(layer(x))
            # maybe add batch norm here
        s_0 = self.output_layer(x)
        return s_0
