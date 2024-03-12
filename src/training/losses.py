import torch.nn as nn
import torch
from typing import Optional


class Loss():
    """
    Class to compute the aggregated loss over rewards, returns and policies.

    Attrs:
    - lr: Optional[nn.Module] = nn.MSELoss() : loss for rewards
    - lv: Optional[nn.Module] = nn.MSELoss() : loss for returns
    - lp: Optional[nn.Module] = nn.CrossEntropyLoss() : loss for policies
    """

    def __init__(self, 
                lr: Optional[nn.Module] = nn.MSELoss(),
                lv: Optional[nn.Module] = nn.MSELoss(), 
                lp: Optional[nn.Module] = nn.CrossEntropyLoss()):
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
        """
        total_loss = torch.tensor(0,dtype=torch.float32 , requires_grad=True)
        M = target_reward.shape[0]

        # Compute the loss for each variable
        for m in range(M):
            current_k = target_horizon[m]
            for i in range(current_k):
                reward_loss = self.lr(pred_reward[m,i], target_reward[m,i])
                return_loss = self.lv(pred_return[m,i], target_return[m,i])
                policy_loss = self.lp(pred_policy[m,i], target_policy[m,i])
                # total_loss = torch.cat([total_loss, reward_loss + return_loss + policy_loss])
                total_loss = torch.add(total_loss, reward_loss + return_loss + policy_loss)

        # total_loss = total_loss.sum()
        
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

