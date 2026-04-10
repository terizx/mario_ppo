
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MarioNet(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        c, h, w = input_shape

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, c, h, w)).shape[1]

        self.fc     = nn.Sequential(layer_init(nn.Linear(cnn_out_dim, 512)), nn.ReLU())
        self.actor  = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)
        x = x.contiguous() / 255.0
        f = self.fc(self.cnn(x))
        return self.actor(f), self.critic(f)


class PPOAgent:
    def __init__(self, input_shape, action_dim,
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2,
                 ent_coef=0.02,
                 ent_target=0.5,
                 vf_coef=0.25,
                 device="cpu"):
        self.device     = device
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef  = clip_coef
        self.ent_coef   = ent_coef
        self.ent_target = ent_target
        self.vf_coef    = vf_coef

        self.network   = MarioNet(input_shape, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            logits, value = self.network(state)
            dist     = Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def update(self, rollouts, batch_size=64):
        states        = torch.FloatTensor(rollouts['states']).to(self.device)
        actions       = torch.LongTensor(rollouts['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
        returns       = torch.FloatTensor(rollouts['returns']).to(self.device)
        advantages    = torch.FloatTensor(rollouts['advantages']).to(self.device)
        old_values    = torch.FloatTensor(rollouts['values']).to(self.device)

        
        ret_mean, ret_std = returns.mean(), returns.std()
        returns    = (returns    - ret_mean) / (ret_std + 1e-8)
        old_values = (old_values - ret_mean) / (ret_std + 1e-8)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_size          = states.shape[0]
        total_policy_loss   = 0
        total_value_loss    = 0
        total_entropy       = 0
        total_approx_kl     = 0
        total_clip_fraction = 0
        num_batches         = 0

        for _ in range(2):
            indices = torch.randperm(total_size)
            for start in range(0, total_size, batch_size):
                idx = indices[start: start + batch_size]

                logits, values = self.network(states[idx])
                dist           = Categorical(logits=logits)
                new_log_probs  = dist.log_prob(actions[idx])
                entropy        = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                surr1       = ratio * advantages[idx]
                surr2       = torch.clamp(ratio, 1 - self.clip_coef,
                                          1 + self.clip_coef) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_unclipped = (values.squeeze() - returns[idx]) ** 2
                v_clipped   = old_values[idx] + torch.clamp(
                    values.squeeze() - old_values[idx],
                    -self.clip_coef, self.clip_coef)
                value_loss  = 0.5 * torch.max(
                    v_unclipped, (v_clipped - returns[idx]) ** 2).mean()

                
                ent_ratio = max(self.ent_target / (entropy.item() + 1e-8), 1.0)
                eff_ent   = self.ent_coef * min(ent_ratio, 10.0)

                loss = policy_loss + self.vf_coef * value_loss - eff_ent * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl     = (-torch.log(ratio)).mean()
                    clip_fraction = (
                        (ratio > 1 + self.clip_coef).float().mean() +
                        (ratio < 1 - self.clip_coef).float().mean()
                    )

                total_policy_loss   += policy_loss.item()
                total_value_loss    += value_loss.item()
                total_entropy       += entropy.item()
                total_approx_kl     += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                num_batches         += 1

        return {
            'loss':          (total_policy_loss + total_value_loss) / num_batches,
            'policy_loss':   total_policy_loss  / num_batches,
            'value_loss':    total_value_loss   / num_batches,
            'entropy':       total_entropy      / num_batches,
            'approx_kl':     total_approx_kl    / num_batches,
            'clip_fraction': total_clip_fraction / num_batches,
        }

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
