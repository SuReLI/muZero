import pytorch_lightning as pl
import torch.nn as nn
import torch
from typing import Optional, List
from training import train_models_one_step

class Dynamics(nn.Module):
    # g dynamics function
    def __init__(self,
                state_dim: int,
                layer_count: Optional[int]=4,
    ):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim

        self.input_layer = nn.Linear(state_dim + 1, state_dim + 1)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(state_dim + 1, state_dim + 1) for _ in range(layer_count)])
        self.output_layer1 = nn.Linear(state_dim + 1, state_dim)
        self.output_layer2 = nn.Linear(state_dim + 1, 1)

    def forward(self, s, a):
        y = torch.concat(s, a)
        y = torch.relu(self.input_layer(y))
        for layer in self.hidden_layers:
            y = torch.relu(layer(y))
            # maybe add batch norm here
        s_k = self.output_layer1(y)
        r_k = self.output_layer2(y)
        return (s_k, r_k)
    
class Prediction(nn.Module):
    # f prediction function
    def __init__(self,
                 state_dim: int,
                 policy_dim: Optional[int] = 1,
                 value_dim: Optional[int] = 1,
                 layer_count: Optional[int]=4,):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim
        self.policy_dim = policy_dim
        self.value_dim = value_dim

        input_dim = self.state_dim
        hidden_dims = self.state_dim # TODO: Experiment with different networks
        output_dim = self.policy_dim + self.value_dim

        self.input_layer = nn.Linear(input_dim, hidden_dims)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims, hidden_dims) for _ in range(layer_count)])
        self.output_layer1 = nn.Linear(hidden_dims, 1)
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
    # h representation function (input: observation, output: state)
    def __init__(self,
                 observation_dim: int,
                 state_dim: int,
                 layer_count: Optional[int]=4,):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim
        self.observation_dim = observation_dim

        input_dim = self.observation_dim
        hidden_dims = self.state_dim
        output_dim = self.state_dim

        self.input_layer = nn.Linear(input_dim, hidden_dims)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims, hidden_dims) for _ in range(layer_count)])
        self.output_layer = nn.Linear(hidden_dims, output_dim)

    def forward(self, o_0):
        x = torch.relu(self.input_layer(o_0))
        for layer in self.hidden_layers:
            y = torch.relu(layer(x))
            # maybe add batch norm here
        s_0 = self.output_layer(x)
        return s_0


class MuModel(pl.LightningModule):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            state_dim: int,
            K: int,
            lr: Optional[float]=0.001,
            layer_count: Optional[int]=4,
            layer_dim: Optional[int]=64,):
        super().__init__()

        self.lf = lr
        self.layer_count = layer_count
        self.layer_dim = layer_dim
        self.K = K
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.automatic_optimization = False # necessary since we define optimizers outside

        self.g = Dynamics(state_dim=state_dim, layer_count=4)
        self.f = Prediction(state_dim=state_dim, policy_dim=1, value_dim=1, layer_count=4)
        self.h = Representation(observation_dim=observation_dim, state_dim=state_dim, layer_count=4)


    def forward(self, x):
        return self.f(x), self.g(x), self.h(x)

    def training_step(self, batch, batch_idx):
        observations, target_actions, target_rewards, target_returns, target_policies = batch
        f_opt, g_opt, h_opt = self.optimizers()

        loss = train_models_one_step(
            observations,
            target_actions,
            target_rewards,
            target_returns,
            target_policies,
            f_opt,
            g_opt,
            h_opt,
            self.h, self.g, self.f,
            verbose=False
        )
        self.log('train_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        f_opt = torch.optim.Adam(self.f.parameters(), lr=self.lf)
        g_opt = torch.optim.Adam(self.g.parameters(), lr=self.lf)
        h_opt = torch.optim.Adam(self.h.parameters(), lr=self.lf)
        return f_opt, g_opt, h_opt