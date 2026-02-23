from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch])
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch])
        d = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buf)
