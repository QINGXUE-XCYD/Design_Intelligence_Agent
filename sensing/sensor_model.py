from __future__ import annotations
import random
from typing import List

from environment.cell import CellType
from environment.grid_map import GridMap, Position
from sensing.observation import Observation


class SensorModel:
    """
    传感器模型 / Sensor model.

    Default behaviour is deterministic fixed-radius sensing. Optional
    false-positive and false-negative rates can be enabled for robustness
    experiments.
    """

    def __init__(
        self,
        range_radius: int,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.range_radius = range_radius
        self.false_positive_rate = max(0.0, min(1.0, false_positive_rate))
        self.false_negative_rate = max(0.0, min(1.0, false_negative_rate))
        self.rng = random.Random(seed)

    def sense(self, env_map: GridMap, robot_pos: Position) -> Observation:
        visible = self.get_visible_cells(env_map, robot_pos)
        obs = Observation()
        obs.visible_cells = visible

        for pos in visible:
            cell_type = env_map.get_cell_type(pos)
            perceived_type = self._apply_noise(cell_type)
            if perceived_type == CellType.FREE:
                obs.free_cells.append(pos)
                if pos in env_map.charging_stations:
                    obs.charging_cells.append(pos)
            elif perceived_type == CellType.STATIC_OBSTACLE:
                obs.occupied_cells.append(pos)
            elif perceived_type == CellType.DYNAMIC_OBSTACLE:
                obs.dynamic_cells.append(pos)
        return obs

    def _apply_noise(self, cell_type: CellType) -> CellType:
        if cell_type == CellType.FREE and self.rng.random() < self.false_positive_rate:
            return CellType.STATIC_OBSTACLE
        if cell_type in (CellType.STATIC_OBSTACLE, CellType.DYNAMIC_OBSTACLE):
            if self.rng.random() < self.false_negative_rate:
                return CellType.FREE
        return cell_type

    def get_visible_cells(self, env_map: GridMap, robot_pos: Position) -> List[Position]:
        rx, ry = robot_pos
        visible: List[Position] = []
        for x in range(env_map.width):
            for y in range(env_map.height):
                if abs(x - rx) + abs(y - ry) <= self.range_radius:
                    visible.append((x, y))
        return visible
