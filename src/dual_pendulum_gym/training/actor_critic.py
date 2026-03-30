import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=14, n_actions=3, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def act(self, obs):
        """Select action from observation (numpy array). Returns action, log_prob, value."""
        x = torch.FloatTensor(obs).unsqueeze(0)
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def evaluate(self, obs_batch, action_batch):
        """Evaluate a batch of observations and actions."""
        logits, values = self(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy
