import pytorch_lightning as pl
import torch.nn as nn
import torch
from typing import Optional, List
from training import train_models_one_step

class Dynamics(nn.Module):
    # g dynamics function
    def __init__(self,
                 layer_count: int,
                 state_dim: int):
        super().__init__()
        self.layer_count = layer_count
        self.state_dim = state_dim

        self.input_layer = nn.Linear(state_dim + 1, state_dim + 1)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(state_dim + 1, state_dim + 1) for _ in range(layer_count)])
        self.output_layer1 = nn.Linear(state_dim + 1, state_dim)
        self.output_layer2 = nn.Linear(state_dim + 1, 1)

    def forward(self, x, a):
        y = torch.concat(x, a)
        y = torch.relu(self.input_layer(y))
        for layer in self.hidden_layers:
            y = torch.relu(layer(y))
            # maybe add batch norm here
        s = self.output_layer1(y)
        r = self.output_layer2(y)
        return (s, r)


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

        self.g = Dynamics(layer_count, observation_dim, state_dim)
        # TODO: prediction and representation
        # self.f = Prediction(...)
        # self.h = Representation(...)


    def forward(self, x):
        return self.g(x), # self.f(x), self.h(x)

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