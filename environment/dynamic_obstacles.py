from __future__ import annotations
import random
from typing import Iterable, Set

from environment.grid_map import GridMap, Position


class RandomWalkDynamicObstaclePolicy:
    """
    简单动态障碍策略 / Simple random-walk dynamic obstacle policy.

    Dynamic obstacles are optional and disabled when count == 0. They avoid
    static obstacles, charging stations, and current robot positions.
    """

    def __init__(self, count: int, seed: int, move_probability: float = 0.30) -> None:
        self.count = max(0, count)
        self.rng = random.Random(seed)
        self.move_probability = max(0.0, min(1.0, move_probability))
        self.initialized = False

    def initialize(self, env_map: GridMap, forbidden: Iterable[Position] = ()) -> None:
        if self.initialized or self.count <= 0:
            return
        forbidden_set = set(forbidden) | env_map.charging_stations
        candidates = [
            p for p in env_map.get_cleanable_cells()
            if p not in forbidden_set and p not in env_map.dynamic_obstacles
        ]
        self.rng.shuffle(candidates)
        env_map.dynamic_obstacles = set(candidates[: self.count])
        self.initialized = True

    def step(self, env_map: GridMap, forbidden: Iterable[Position] = ()) -> None:
        if self.count <= 0:
            return
        self.initialize(env_map, forbidden)
        forbidden_set: Set[Position] = set(forbidden) | env_map.charging_stations
        old_obstacles = list(env_map.dynamic_obstacles)
        new_obstacles: Set[Position] = set()
        env_map.dynamic_obstacles.clear()

        for pos in old_obstacles:
            new_pos = pos
            if self.rng.random() < self.move_probability:
                candidates = [pos] + env_map.get_neighbors(pos)
                self.rng.shuffle(candidates)
                for candidate in candidates:
                    if (
                        env_map.in_bounds(candidate)
                        and candidate not in env_map.static_obstacles
                        and candidate not in forbidden_set
                        and candidate not in new_obstacles
                    ):
                        new_pos = candidate
                        break
            if new_pos not in forbidden_set and new_pos not in env_map.static_obstacles:
                new_obstacles.add(new_pos)

        env_map.dynamic_obstacles = new_obstacles
