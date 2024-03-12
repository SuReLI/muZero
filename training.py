"""
File in which we define the training loop for the model.
"""

import torch
import torch.nn as nn


class LossMuZero():
    def __init__(self, lr, lv, lp):
        self.lr = lr
        self.lv = lv
        self.lp = lp

    def compute_loss(
            self, 
            target_reward, 
            pred_reward,  
            target_return, 
            pred_return, 
            target_policy, 
            pred_policy):
        """
        Compute the loss of the model, given targets and predictions for 
        the next K unrolled steps.

        Args:
        - target_reward: tensor of target rewards     [M*K*1]
        - pred_reward  : tensor of predicted rewards  [M*K*1]
        - target_return: tensor of target returns     [M*K*1]
        - pred_return  : tensor of predicted returns  [M*K*1]
        - target_policy: tensor of target policies    [M*K*A] (density)
        - pred_policy  : tensor of predicted policies [M*K*A] (density)

        Loss on the reward is : MSE
        Loss on the return is : MSE
        Loss on the policy is : Cross-entropy
        """
        total_loss = 0
        K = target_reward.shape[1]

        # Compute the loss for each variable
        for i in range(K):
            reward_loss = self.lr(pred_reward[i], target_reward[i])
            return_loss = self.lv(pred_return[i], target_return[i])
            policy_loss = self.lp(pred_policy[i], target_policy[i])
            total_loss += reward_loss + return_loss + policy_loss

        return total_loss
    

def compute_predictions(
    observations,
    target_actions,
    h, g, f, 
    horizon
):
    """
    Computes the predictions for each network. Works on batches.

    Args:
    - observations: tensor of observations       [M*N*O]
    - target_actions: tensor of target actions   [M*K*A]
    - target_horizon: number of unrolled steps for each trajectory [M*1]
    - h: model for the representation (obs[N-tuple] -> hidden state 's0')
    - g: model for the dynamics (reward, state)
    - f: model for the prediction (policy, value)
    - horizon: number of unrolled steps (ideal: K=5)
    """
    # Initialize the prediction lists
    pred_rewards = []
    pred_returns = []
    pred_policies = []

    # Predictions for the next K unrolled steps
    prev_state = h(observations) 

    for i in range(horizon):
        # Predict the reward and the next state
        reward, state = g(prev_state, target_actions[:,i])

        # Predict the policy and the value
        policy, value = f(state)

        # Store the predictions
        pred_rewards.append(reward)  # List of K elements of size [M]
        pred_returns.append(value)
        pred_policies.append(policy)

        # Update the previous state
        prev_state = state

    # Reshape the predictions [M*K]
    pred_rewards  = torch.stack(pred_rewards, dim=1)
    pred_returns  = torch.stack(pred_returns, dim=1)
    pred_policies = torch.stack(pred_policies, dim=1)

    # Apply mask over trajectories
    # 1 * target_horizon + 0 * reste
    

    return pred_rewards, pred_returns, pred_policies


def train_models_one_step(
    observations,
    target_actions,
    target_rewards,
    target_returns,
    target_policies,
    optimizer_h,
    optimizer_g,
    optimizer_f,
    criterion,
    h, g, f,
    horizon,
    verbose=False
):
    """
    Performs one-step gradient descent on the models. Works on batches.
    
    Args:
    - observations: tensor of observations       [M*N*O]
    - target_actions: tensor of target actions   [M*K*A] (one-hot) (A=nb_actions)
    - target_rewards: tensor of target rewards   [M*K*1]
    - target_returns: tensor of target returns   [M*K*1]
    - target_policies: tensor of target policies [M*K*A] (density)
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
    preds = compute_predictions(observations, target_actions, h, g, f, horizon)
    pred_rewards, pred_returns, pred_policies = preds

    # Compute the loss
    loss = criterion.compute_loss(
        target_rewards, 
        pred_rewards,
        target_returns, 
        pred_returns,
        target_policies, 
        pred_policies
    )
    loss.backward()

    # Update the models
    optimizer_h.step()
    optimizer_g.step()
    optimizer_f.step()

    # Print if needed
    if verbose:
        print(f"Loss: {loss.item():.4f}")

    return loss.item()


def valid_models_one_step(
    observations,
    target_actions,
    target_rewards,
    target_returns,
    target_policies,
    criterion,
    h, g, f,
    horizon,
    verbose=False
):
    """
    Performs one-step gradient descent on the models. Works on batches.
    
    Args:
    - observations: tensor of observations       [M*N*O]
    - target_actions: tensor of target actions   [M*K*A] (one-hot) (A=nb_actions)
    - target_rewards: tensor of target rewards   [M*K*1]
    - target_returns: tensor of target returns   [M*K*1]
    - target_policies: tensor of target policies [M*K*A] (density)
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
            target_rewards, pred_rewards,
            target_returns, pred_returns,
            target_policies, pred_policies
        )

    # Print if needed
    if verbose:
        print(f"Loss: {loss.item():.4f}")

    return loss.item()

