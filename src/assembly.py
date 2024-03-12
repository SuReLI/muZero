from typing import List
import os

import torch
from torch import Tensor
from torch.nn import Module

from environment import AbstractEnvironment, DummyEnvironment
from replaybuffer import ReplayBuffer

from training.mu_model import MuModel


def planning(
    representation: Module,
    dynamics: Module,
    prediction: Module,
    o: Tensor,
    mask: Tensor,
) -> tuple[float, Tensor]:
    pass


def acting(
    env: AbstractEnvironment,
    representation: Module,
    dynamics: Module,
    prediction: Module,
) -> List[tuple[Tensor, Tensor, int, float, float]]:
    pass


def training(
    observations: Tensor,
    target_policies: Tensor,
    target_actions: Tensor,
    target_returns: Tensor,
    episode_lengths: Tensor,
    representation: Module,
    dynamics: Module,
    prediction: Module,
) -> float:
    pass


def h(o: Tensor) -> Tensor:
    pass


def g(s: Tensor, a: Tensor) -> tuple[float, Tensor]:
    pass


def f(s: Tensor) -> tuple[float, Tensor]:
    pass


def main(
    initital_exploration: int,
    replay_buffer_capacity: int = 10_000,
    minibatch_size: int = 64,
    minibatch_nb: int = 1_000,
    exploration_every: int = 5,
    exploration_size: int = 500,
    look_ahead_steps: int = 5,
    look_back_steps: int = 10,
    save_models_every: int | None = 5,
    save_models_to: str = "models",
    verbose: bool = True,
):
    os.makedirs(save_models_to, exist_ok=True)

    rb = ReplayBuffer(capacity=replay_buffer_capacity)
    env = DummyEnvironment()
    
    # Initialize the MuModel
    # [?] Supposing that env.observation_space.n is the dimension of the observation space and 
    #     env.action_space.n is the number of possible actions (gym environment specific)
    mu_model = MuModel(
        observation_dim=env.observation_space.n, # dimension of the observation space (Cart Pole: 4)
        action_dim=env.action_space.n, # number of possible actions (Cart Pole: 2)
        N=look_back_steps, # number of past observations used during training (arbitrary)
        K=look_ahead_steps, # number of future steps used during training (arbitrary)
        state_dim=look_back_steps * env.observation_dim, # dimension of the state space (arbitrary)
    )

    # Initial exploration to fill in the replay buffer
    for _ in range(initital_exploration):
        episode = acting(env, h, g, f)
        rb.push(episode)

    for i in range(minibatch_nb):
        (
            observations,
            target_policies,
            target_actions,
            # [!] target_reward missing
            target_returns,
            episode_lengths,
        ) = rb.sample(minibatch_size,
                      LOOK_AHEAD_STEPS=look_ahead_steps,
                      LOOK_BACK_STEPS=look_back_steps)

        loss = mu_model.training_step(
            (
                observations,
                target_policies,
                target_actions,
                # [!] target_reward missing
                target_returns,
                episode_lengths,)
        )

        if verbose:
            print(f"Minibatch {i}: loss= {loss}")

        if (i + 1) % exploration_every:
            for _ in range(exploration_size):
                episode = acting(env, h, g, f)
                rb.push(episode)

        if save_models_every is not None and (i + 1) % save_models_every:
            torch.save(f.state_dict(), f"{save_models_to}/prediction_{i}.pt")
            torch.save(g.state_dict(), f"{save_models_to}/dynamics_{i}.pt")
            torch.save(h.state_dict(), f"{save_models_to}/representation_{i}.pt")
