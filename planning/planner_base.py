from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

Position = Tuple[int, int]


class PlannerBase(ABC):
    """
    路径规划器抽象基类 / Abstract base class for path planners
    """

    @abstractmethod
    def plan(self, start: Position, goal: Position, belief_map) -> List[Position]:
        """
        从 start 规划到 goal 的路径 / Plan a path from start to goal
        """
        raise NotImplementedError