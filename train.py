import numpy as np
import torch
import torch.optim as optim
import copy
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
import argparse
from data.dataset import Dataset
from models.voice_transformer_moe import VoiceMoETransformer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- Training Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Shared Helper Functions ---
def sample_actions(logits, num_samples):
    """
    Samples a specified number of actions based on a probability distribution derived from logits.

    Args:
        logits (torch.Tensor): The input logits from which to derive the probability distribution.
        num_samples (int): The number of actions to sample from the distribution.

    Returns:
        tuple: A tuple containing:
            - actions (torch.Tensor): A tensor of sampled actions.
            - probs (torch.Tensor): The softmax probabilities corresponding to the input logits.
    """

    probs = F.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, num_samples, replacement=True)
    return actions, probs

def compute_rewards(actions, labels):
    """
    Computes binary rewards from sampled actions and their corresponding labels.

    Args:
        actions (torch.Tensor): A tensor of shape (batch_size, num_samples) containing sampled actions.
        labels (torch.Tensor): A tensor of shape (batch_size) containing the corresponding labels.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_samples) containing the binary rewards.
    """
    labels_exp = labels.unsqueeze(1).expand_as(actions)
    return (actions == labels_exp).float()

def compute_group_advantages(rewards, std_eps=1e-8):
    """
    Computes group relative advantages from rewards.

    Given a tensor of rewards, computes the group relative advantages by subtracting the
    group mean and dividing by the group standard deviation. A small epsilon value is
    added to the standard deviation to prevent division by zero.

    Args:
        rewards (torch.Tensor): A tensor of shape (batch_size, num_samples) containing rewards.
        std_eps (float, optional): A small value added to the standard deviation to prevent division by zero.
            Defaults to 1e-8.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_samples) containing the group relative advantages.
    """
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True) + std_eps
    return (rewards - mean_r) / std_r

def compute_kl(old_probs, current_probs):
    """
    Computes the KL divergence between the old and current policy distributions.

    Args:
        old_probs (torch.Tensor): A tensor of shape (batch_size, num_actions) containing the old policy probabilities.
        current_probs (torch.Tensor): A tensor of shape (batch_size, num_actions) containing the current policy probabilities.

    Returns:
        float: The mean KL divergence between the old and current policy distributions.
    """
    kl = (old_probs * (old_probs.log() - current_probs.log())).sum(dim=1)
    return kl.mean()

def train_ppo_model(model_class, epochs=200, clip_epsilon=0.2, entropy_coef=0.01, **model_kwargs):
    """
    Trains a model using the Proximal Policy Optimization (PPO) algorithm.

    This function trains a given model class using the PPO algorithm, which
    involves taking snapshots of the model, computing logits and probabilities
    for actions, and optimizing based on a surrogate loss that incorporates
    policy and entropy losses.

    Args:
        model_class (type): The class of the model to be trained.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        clip_epsilon (float, optional): Clipping parameter for the PPO objective. Defaults to 0.2.
        entropy_coef (float, optional): Coefficient for the entropy term. Defaults to 0.01.
        **model_kwargs: Additional keyword arguments to initialize the model.

    Returns:
        model: The trained model.
    """

    model = model_class(**model_kwargs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Snapshot old model
            old_model = copy.deepcopy(model)
            old_model.eval()
            
            optimizer.zero_grad()
            with torch.no_grad():
                old_logits = old_model(data)
            current_logits = model(data)
            
            old_probs = F.softmax(old_logits, dim=-1)
            actions = torch.multinomial(old_probs, 1).squeeze(1)
            rewards = (actions == target).float()
            
            current_probs = F.softmax(current_logits, dim=-1)
            current_probs_actions = current_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            old_probs_actions = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = current_probs_actions / (old_probs_actions + 1e-8)
            
            advantages = rewards  # Simple advantage computation
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = -(current_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()
            loss = policy_loss - entropy_coef * entropy
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        
        # Evaluation
        model.eval()
        all_targets, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                all_targets.append(target.cpu())
                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()[:, 1]
        
        test_acc = 100.0 * (all_preds == all_targets).mean()
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        print(f'Epoch {epoch+1}/{epochs} | PPO Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Test F1: {f1:.4f} | Test AUC: {auc:.4f}')
        model.train()
    return model

# --- GRPO Training (Generic for any model) ---
def train_grpo_model(model_class, epochs=200, group_size=20, clip_epsilon=0.2, kl_coef=0.01, **model_kwargs):
    """
    Train a model using the Generalized Reward Policy Optimization (GRPO) algorithm.

    Args:
        model_class: A class implementing the model architecture to be trained.
        epochs: The number of epochs to train.
        group_size: The size of the groups used in the GRPO algorithm.
        clip_epsilon: The clipping value used for the policy ratio.
        kl_coef: The coefficient of the KL divergence term in the loss.
        **model_kwargs: Additional keyword arguments to be passed to the model class.

    Returns:
        The trained model.
    """
    model = model_class(**model_kwargs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        old_model = copy.deepcopy(model)
        old_model.eval()
        
        total_loss = 0.0
        total_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            logits_current = model(data)
            with torch.no_grad():
                logits_old = old_model(data)
            
            actions, old_probs_full = sample_actions(logits_old, group_size)
            current_probs_full = F.softmax(logits_current, dim=-1)
            p_old = torch.gather(old_probs_full, 1, actions)
            p_current = torch.gather(current_probs_full, 1, actions)
            
            rewards = compute_rewards(actions, target)
            advantages = compute_group_advantages(rewards)
            
            ratio = p_current / (p_old + 1e-8)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            surrogate_loss = torch.where(advantages >= 0, torch.min(unclipped, clipped), torch.max(unclipped, clipped))
            loss_policy = -surrogate_loss.mean()
            kl_loss = compute_kl(old_probs_full, current_probs_full)
            loss = loss_policy + kl_coef * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        
        # Evaluation
        model.eval()
        all_targets, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                all_targets.append(target.cpu())
                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()[:, 1]
        
        test_acc = 100.0 * (all_preds == all_targets).mean()
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        print(f'Epoch {epoch+1}/{epochs} | GRPO Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Test F1: {f1:.4f} | Test AUC: {auc:.4f}')
        model.train()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VoiceGRPO Training')
    parser.add_argument('--train',  type=str, default='grpo', help='Train a new model')
    parser.add_argument('--epoch',  type=int, default=100, help='Number of epochs')
    parser.add_argument('--dataset',  type=str, default='./data/synthetic_pathology_dataset.V.1.0.xlsx', help='Set dataset path')
    
    args = parser.parse_args()

    dataset = Dataset(dataset_path=args.dataset)
    train_dataset, test_dataset = dataset.prepare()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    epoch = args.epoch

    if args.train=='grpo':
       trained_moe_grpo = train_grpo_model(VoiceMoETransformer, epochs=epoch, input_dim=6, num_classes=2, dim=64, depth=3, heads=4, num_experts=4)
  
    elif args.train=='ppo':
       trained_moe_ppo = train_ppo_model(VoiceMoETransformer, epochs=epoch, input_dim=6, num_classes=2, dim=64, depth=3, heads=4, num_experts=3)
 
