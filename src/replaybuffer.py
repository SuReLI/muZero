from typing import List
from collections import namedtuple
import random

import torch
from torch import Tensor


### TYPES ###
EpisodeStep = namedtuple(
    "EpisodeStep", ["observation", "policy", "action", "reward", "value"]
)


### REPLAY BUFFER ###
class ReplayBuffer:
    """
    First-in first-out replay buffer for storing episodes.

    The first `capacity` episodes are stored in the buffer. When the buffer is
    full, the oldest episode is overwritten by the newest episode.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Initialize an empty first-in first-out replay buffer.

        Parameters
        ----------
        * `capacity`: capacity of the replay buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, episode: List[EpisodeStep]):
        """
        Insert a new episode into the replay buffer.

        Parameters
        ----------
        * `episode`: the episode to be inserted into the replay buffer. List of
            `EpisodeStep` named tuples `(observation: Tensor, policy: Tensor,
            action: int, reward: float, value: float)`.
        """

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, LOOK_AHEAD_STEPS: int = 5, LOOK_BACK_STEPS=10
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieve a random batch of episodes from the replay buffer.

        Parameters
        ----------
        * `batch_size`: number of episodes to sample from the replay buffer.

        Returns
        -------
        * `observations`
        * `policies`
        * `actions`
        * `rewards`
        * `values`
        * `episode_lengths`

        Throws
        ------
        * `ValueError`: if the replay buffer is empty or smaller than the
            requested batch size.
        """

        if len(self.buffer) < batch_size:
            raise ValueError(
                "Replay buffer is empty or smaller than the requested batch size"
            )

        sampled_episodes = random.sample(self.buffer, batch_size)
        start_points = [random.randint(0, len(episode) - 1) for episode in sampled_episodes]

        observations_batch = []
        policies_batch = []
        actions_batch = []
        rewards_batch = []
        values_batch = []
        episode_lengths = []

        for episode, start_point in zip(sampled_episodes, start_points):
            # Transform the episode into a tuple of lists
            episode = tuple(map(list, zip(*episode)))
            observations, policies, actions, rewards, values = episode

            # Parameters
            length = len(observations)
            state_space_size = observations[0].size()
            action_space_size = policies[0].size(0)

            # Pad observations at the beginning with zeros
            observations = torch.cat(
                [
                    torch.zeros(
                        (max(0, LOOK_BACK_STEPS - 1 - start_point), *state_space_size)
                    ),
                    torch.stack(
                        observations[
                            max(0, start_point - LOOK_BACK_STEPS + 1) : start_point + 1
                        ]
                    ),
                ]
            )
            observations_batch.append(observations)

            # Pad policies with zeros at the end
            policies = torch.concat(
                [
                    torch.stack(policies[start_point : start_point + LOOK_AHEAD_STEPS + 1]),
                    torch.zeros(
                        (
                            max(0, LOOK_AHEAD_STEPS + 1 - length + start_point),
                            action_space_size,
                        )
                    ),
                ]
            )
            policies_batch.append(policies)

            # Pad actions with tensor of zeros at the end
            actions = torch.concat(
                [
                    torch.stack(actions[start_point : start_point + LOOK_AHEAD_STEPS + 1]),
                    torch.zeros(
                        (
                            max(0, LOOK_AHEAD_STEPS + 1 - length + start_point),
                            action_space_size,
                        )
                    ),
                ]
            )
            actions_batch.append(actions)

            # Pad rewards with the last reward at the end
            rewards = torch.concat(
                [
                    torch.tensor(rewards[start_point : start_point + LOOK_AHEAD_STEPS + 1]),
                    torch.full(
                        (max(0, LOOK_AHEAD_STEPS + 1 - length + start_point),), rewards[-1]
                    ),
                ]
            )
            rewards_batch.append(rewards)

            # Pad values with zeros at the end
            values = torch.concat(
                [
                    torch.tensor(values[start_point : start_point + LOOK_AHEAD_STEPS + 1]),
                    torch.zeros((max(0, LOOK_AHEAD_STEPS + 1 - length + start_point),)),
                ]
            )
            values_batch.append(values)

            episode_lengths.append(min(LOOK_AHEAD_STEPS, length - start_point))

        return (
            torch.stack(observations_batch),
            torch.stack(policies_batch),
            torch.stack(actions_batch),
            torch.stack(rewards_batch),
            torch.stack(values_batch),
            torch.tensor(episode_lengths),
        )
