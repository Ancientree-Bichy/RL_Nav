from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RLAgent(ABC):
    def start_episode(self) -> None:
        return None

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        raise NotImplementedError

    def end_episode(self) -> dict[str, float]:
        return {}

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, payload: dict) -> None:
        raise NotImplementedError
