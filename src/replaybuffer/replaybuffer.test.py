import numpy as np
import torch

from replaybuffer import ReplayBuffer, EpisodeStep


def __generate_random_episode(
    episode_length: int, observation_size: int, action_size: int
):
    """
    Generate a random episode of a given length.

    Parameters
    ----------
    * `episode_length`: length of the episode

    Returns
    -------
    * `episode`: the generated episode. List of `EpisodeStep` named tuples
        `(observation: Tensor, policy: Tensor, action: int, reward: float,
        value: float)`.
    """

    episode = []
    for _ in range(episode_length):
        observation = torch.randn(observation_size)
        policy = torch.randn(action_size)
        action = np.random.randint(2)
        reward = np.random.rand()
        value = np.random.rand()
        episode.append(EpisodeStep(observation, policy, action, reward, value))
    return episode


def test_replay_buffer():
    O = 12
    A = 2
    N = 10
    K = 5
    M = 4
    EP_LENGTH = 25

    rb = ReplayBuffer(capacity=20)

    # Push 10 episodes into the replay buffer
    for _ in range(10):
        ep = __generate_random_episode(EP_LENGTH, O, A)
        rb.push(ep)

    # Check that the replay buffer has 10 episodes
    assert len(rb) == 10

    # Sample 5 episodes from the replay buffer
    observations, policies, actions, rewards, values, lengths = rb.sample(
        M, LOOK_AHEAD_STEPS=K, LOOK_BACK_STEPS=N
    )

    # Check that the sampled episodes have the correct shape
    assert observations.size() == (M, N, O), f"observations.size() = {observations.size()}"
    assert policies.size() == (M, K + 1, A), f"policies.size() = {policies.size()}"
    assert actions.size() == (M, K + 1), f"actions.size() = {actions.size()}"
    assert rewards.size() == (M, K + 1), f"rewards.size() = {rewards.size()}"
    assert values.size() == (M, K + 1), f"values.size() = {values.size()}"
    assert lengths.size() == (M,), f"lengths.size() = {lengths.size()}"

    # Fill the replay buffer to the max capacity
    for _ in range(10):
        ep = __generate_random_episode(EP_LENGTH, O, A)
        rb.push(ep)

    ep = __generate_random_episode(EP_LENGTH, O, A)
    rb.push(ep)

    assert rb.buffer[0] == ep
    assert len(rb) == 20


if __name__ == "__main__":
    test_replay_buffer()
    print("assembly.py all tests passed")
