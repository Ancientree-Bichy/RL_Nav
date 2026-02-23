from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlnav.networks import MLP

from .base import RLAgent


class PolicyGradientAgent(RLAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
    ):
        self.device = device
        self.gamma = gamma

        self.policy = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []

    def start_episode(self) -> None:
        self._states = []
        self._actions = []
        self._rewards = []

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(st)
        dist = torch.distributions.Categorical(logits=logits)

        if explore:
            return int(dist.sample().item())

        probs = dist.probs.squeeze(0)
        return int(torch.argmax(probs).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        del next_state, done
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        return {}

    def end_episode(self) -> dict[str, float]:
        if not self._rewards:
            return {}

        returns = []
        g = 0.0
        for r in reversed(self._rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()

        g_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        g_t = (g_t - g_t.mean()) / (g_t.std() + 1e-8)

        st_batch = torch.tensor(np.stack(self._states), dtype=torch.float32, device=self.device)
        a_batch = torch.tensor(self._actions, dtype=torch.int64, device=self.device)

        logits = self.policy(st_batch)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a_batch)

        loss = -(logp * g_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
