import os
from typing import List

import torch

from acting import acting
from environment import GymWrapper
from training import MuModel
from replaybuffer import ReplayBuffer


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
    mcts_budget: int = 10,
    verbose: bool = True,
):
    os.makedirs(save_models_to, exist_ok=True)

    rb = ReplayBuffer(capacity=replay_buffer_capacity)

    # Wrap the environment
    env = GymWrapper("CartPole-v1")

    # Initialize the MuModel
    mu_model = MuModel(
        observation_dim=env.observation_space.shape[
            0
        ],  # dimension of the observation space (Cart Pole: 4)
        action_dim=env.action_space.n,  # number of possible actions (Cart Pole: 2)
        N=look_back_steps,  # number of past observations used during training (arbitrary)
        K=look_ahead_steps,  # number of future steps used during training (arbitrary)
        state_dim=look_back_steps
        * env.observation_space.shape[0],  # dimension of the state space (arbitrary)
    )

    # Initial exploration to fill in the replay buffer
    for _ in range(initital_exploration):
        episode = acting(env, mu_model.h, mu_model.g, mu_model.f)
        rb.push(episode)

    # Training loop
    for i in range(minibatch_nb):
        (
            observations,
            target_policies,
            target_actions,
            target_rewards,
            target_returns,
            episode_lengths,
        ) = rb.sample(
            minibatch_size,
            LOOK_AHEAD_STEPS=look_ahead_steps,
            LOOK_BACK_STEPS=look_back_steps,
        )

        loss = mu_model.training_step(
            (
                observations,
                target_policies,
                target_actions,
                target_rewards,
                target_returns,
                episode_lengths,
            )
        )

        if verbose:
            print(f"Minibatch {i}: loss= {loss}")

        if (i + 1) % exploration_every:
            for _ in range(exploration_size):
                episode = acting(env, mu_model.h, mu_model.g, mu_model.f)
                rb.push(episode)

        if save_models_every is not None and (i + 1) % save_models_every:
            torch.save(mu_model.f.state_dict(), f"{save_models_to}/prediction_{i}.pt")
            torch.save(mu_model.g.state_dict(), f"{save_models_to}/dynamics_{i}.pt")
            torch.save(
                mu_model.h.state_dict(), f"{save_models_to}/representation_{i}.pt"
            )


if __name__ == "__main__":
    main(initital_exploration=1_000, verbose=True)
