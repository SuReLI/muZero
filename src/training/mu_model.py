from typing import Optional

import torch
import torch.nn as nn

from .models import Dynamics, Prediction, Representation
from .losses import Loss, compute_predictions
from .optimizers import Optimizers


class MuModel(nn.Module):
    """
    Class in which the three models (h, g, f) are stored. It enables
    training and validation steps.

    Attrs:
    - observation_dim [int] : dimension of the observation space
    - action_dim [int] : dimension of the action space
    - state_dim [int] : dimension of the state space
    - N [int] : number of past observations
    - K [int] : number of unrolled steps
    - lr [float] : learning rate

    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        state_dim: int,
        N: int,
        K: int,
        lr: Optional[float] = 0.001,
        layer_count: Optional[int] = 4,
        layer_dim: Optional[int] = 64,
        criterion: Optional[Loss] = Loss(),
        h_model: Optional[nn.Module] = None,
        g_model: Optional[nn.Module] = None,
        f_model: Optional[nn.Module] = None,
        optimizers: Optional[Optimizers] = None,
    ):
        super().__init__()

        self.lr = lr
        self.layer_count = layer_count
        self.layer_dim = layer_dim
        self.K = K
        self.N = N
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.criterion = criterion

        h_model = h_model if h_model is not None else Representation
        g_model = g_model if g_model is not None else Dynamics
        f_model = f_model if f_model is not None else Prediction

        self.g = g_model(state_dim=state_dim, action_dim=action_dim, layer_count=4)
        self.f = f_model(
            state_dim=state_dim, policy_dim=action_dim, value_dim=1, layer_count=4
        )
        self.h = h_model(
            observation_dim=observation_dim,
            state_dim=state_dim,
            N=self.N,
            layer_count=4,
        )

        self.optimizers = (
            optimizers
            if optimizers is not None
            else Optimizers.basic_optimizer(self.h, self.g, self.f)
        )

    def forward(self, x):
        return self.f(x), self.g(x), self.h(x)

    def _train_models_one_step(
        self,
        observations,
        target_policies,
        target_actions,
        target_rewards,
        target_returns,
        target_horizon,
        optimizer_h,
        optimizer_g,
        optimizer_f,
        criterion,
        h,
        g,
        f,
        horizon,
        verbose=False,
    ):
        """
        Performs one-step gradient descent on the models. Works on batches.

        Args:
        - observations: tensor of observations       [M*N*O]
        - target_actions: tensor of target actions   [M*K*A] (one-hot) (A=nb_actions)
        - target_rewards: tensor of target rewards   [M*K*1]
        - target_returns: tensor of target returns   [M*K*1]
        - target_policies: tensor of target policies [M*K*A] (density)
        - target_horizon: number of unrolled steps for each trajectory [M]
        - optimizer_h: optimizer for the model h
        - optimizer_g: optimizer for the model g
        - optimizer_f: optimizer for the model h
        - criterion: the global loss class (containing the 3 losses)
        - h: model for the representation (obs[N-tuple] -> hidden state 's0')
        - g: model for the dynamics (reward, state)
        - f: model for the prediction (policy, value)
        - horizon: number of unrolled steps (ideal: K=5)s
        - verbose: print the loss at (each) iteration
        """

        # Set gradients to zero
        optimizer_h.zero_grad()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        # Compute the predictions
        pred_rewards, pred_returns, pred_policies = compute_predictions(
            observations, target_actions, h, g, f, horizon
        )

        # Compute the loss
        loss = criterion.compute_loss(
            target_rewards,
            pred_rewards,
            target_returns,
            pred_returns,
            target_policies,
            pred_policies,
            target_horizon,
        )
        loss.backward()

        # Update the models
        optimizer_h.step()
        optimizer_g.step()
        optimizer_f.step()

        # Print if needed
        if verbose:
            print(f"Training Loss: {loss.item():.4f}")

        return loss

    def _valid_models_one_step(
        observations,
        target_policies,
        target_actions,
        target_rewards,
        target_returns,
        target_horizon,
        criterion,
        h,
        g,
        f,
        horizon,
        verbose=False,
    ):
        """
        Performs one-step gradient descent on the models. Works on batches.

        Args:
        - observations: tensor of observations       [M*N*O]
        - target_actions: tensor of target actions   [M*K*A] (one-hot) (A=nb_actions)
        - target_rewards: tensor of target rewards   [M*K*1]
        - target_returns: tensor of target returns   [M*K*1]
        - target_policies: tensor of target policies [M*K*A] (density)
        - target_horizon: number of unrolled steps for each trajectory [M]
        - criterion: the global loss class (containing the 3 losses)
        - h: model for the representation (obs[N-tuple] -> hidden state 's0')
        - g: model for the dynamics (reward, state)
        - f: model for the prediction (policy, value)
        - horizon: number of unrolled steps (ideal: K=5)
        - verbose: print the loss at (each) iteration
        """

        with torch.no_grad():
            # Compute the predictions
            preds = compute_predictions(observations, target_actions, h, g, f, horizon)
            pred_rewards, pred_returns, pred_policies = preds

            # Compute the loss
            loss = criterion.compute_loss(
                target_rewards,
                pred_rewards,
                target_returns,
                pred_returns,
                target_policies,
                pred_policies,
                target_horizon,
            )

        # Print if needed
        if verbose:
            print(f"Validation Loss: {loss.item():.4f}")

        return loss.item()

    def training_step(self, batch):
        (
            observations,
            target_policies,
            target_actions,
            target_rewards,
            target_returns,
            target_horizon,
        ) = batch

        loss = self._train_models_one_step(
            observations=observations,
            target_policies=target_policies,
            target_actions=target_actions,
            target_rewards=target_rewards,
            target_returns=target_returns,
            target_horizon=target_horizon,
            optimizer_h=self.optimizers.opt_h,
            optimizer_g=self.optimizers.opt_g,
            optimizer_f=self.optimizers.opt_f,
            criterion=self.criterion,
            h=self.h,
            g=self.g,
            f=self.f,
            horizon=self.K,
            verbose=False,
        )

        return loss

    def validation_step(self, batch):

        (
            observations,
            target_policies,
            target_actions,
            target_rewards,
            target_returns,
            target_horizon,
        ) = batch

        loss = self._valid_models_one_step(
            observations,
            target_policies,
            target_actions,
            target_rewards,
            target_returns,
            target_horizon,
            self.criterion,
            self.h,
            self.g,
            self.f,
            self.K,
            verbose=True,
        )

        return loss
