from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlnav.infra.replay_buffer import ReplayBuffer, Transition
from rlnav.networks import MLP

from .base import RLAgent


class DQNAgent(RLAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup: int = 500,
        target_update: int = 200,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.999,
    ):
        self.device = device
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update = target_update

        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = eps_start

        self.step_count = 0

        self.q_net = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=buffer_size)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q_net(st), dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        self.step_count += 1
        self.replay.push(Transition(s=state, a=action, r=reward, s2=next_state, done=done))

        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.replay) >= max(self.warmup, self.batch_size):
            loss = self._learn_step()
            metrics["loss"] = loss

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        metrics["epsilon"] = self.epsilon
        return metrics

    def _learn_step(self) -> float:
        bs, ba, br, bs2, bd = self.replay.sample(self.batch_size)

        bs_t = torch.tensor(bs, dtype=torch.float32, device=self.device)
        ba_t = torch.tensor(ba, dtype=torch.int64, device=self.device).unsqueeze(1)
        br_t = torch.tensor(br, dtype=torch.float32, device=self.device).unsqueeze(1)
        bs2_t = torch.tensor(bs2, dtype=torch.float32, device=self.device)
        bd_t = torch.tensor(bd, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q_net(bs_t).gather(1, ba_t)
        with torch.no_grad():
            max_q_next = self.target_net(bs2_t).max(dim=1, keepdim=True).values
            y = br_t + self.gamma * max_q_next * (1.0 - bd_t)

        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.step_count = int(payload.get("step_count", self.step_count))
