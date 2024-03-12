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
            pred_policy, 
            target_horizon):
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
        - target_horizon : number of unrolled steps for each trajectory [M*1]

        Loss on the reward is : MSE
        Loss on the return is : MSE
        Loss on the policy is : Cross-entropy
        """
        total_loss = 0
        M = target_reward.shape[0]

        # Compute the loss for each variable
        for m in range(M):
            current_k = target_horizon[m]
            for i in range(current_k):
                reward_loss = self.lr(pred_reward[m,i], target_reward[m,i])
                return_loss = self.lv(pred_return[m,i], target_return[m,i])
                policy_loss = self.lp(pred_policy[m,i], target_policy[m,i])
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
    - h: model for the representation (obs[N-tuple] -> hidden state 's0')
    - g: model for the dynamics (reward, state)
    - f: model for the prediction (policy, value)
    - horizon: number of unrolled steps (ideal: K=5)
    """
    # Initialize the prediction lists
    preds_reward = []
    preds_return = []
    preds_policy = []

    # Predictions for the next K unrolled steps
    prev_state = h(observations) 

    for i in range(horizon):
        # Predict the reward and the next state
        reward, state = g(prev_state, target_actions[:,i])

        # Predict the policy and the value
        policy, value = f(state)

        # Store the predictions
        preds_reward.append(reward)  # List of K elements of size [M]
        preds_return.append(value)
        preds_policy.append(policy)

        # Update the previous state
        prev_state = state

    # Reshape the predictions [M*K]
    preds_reward = torch.stack(preds_reward, dim=1)
    preds_return = torch.stack(preds_return, dim=1)
    preds_policy = torch.stack(preds_policy, dim=1)

    # Apply mask over trajectories
    # k_tensor = torch.tensor(target_horizon)[:, None] 
    # indices = torch.arange(horizon)[None, :]
    # k_mask = (indices < k_tensor).int()

    # preds_reward = preds_reward * k_mask
    # preds_return = preds_return * k_mask
    # preds_policy = preds_policy * k_mask
    

    return preds_reward, preds_return, preds_policy


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

