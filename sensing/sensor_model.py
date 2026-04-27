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
        mode: str = "manhattan",
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.range_radius = range_radius
        self.mode = mode
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
        if self.mode == "manhattan":
            return self._visible_cells_manhattan(env_map, robot_pos)
        if self.mode == "euclidean":
            return self._visible_cells_euclidean(env_map, robot_pos)
        if self.mode == "occluded_manhattan":
            return self._visible_cells_occluded_manhattan(env_map, robot_pos)
        raise ValueError(
            f"Unsupported sensor mode: {self.mode}. "
            "Expected one of: manhattan, euclidean, occluded_manhattan."
        )

    def _visible_cells_manhattan(self, env_map: GridMap, robot_pos: Position) -> List[Position]:
        rx, ry = robot_pos
        visible: List[Position] = []
        for x in range(env_map.width):
            for y in range(env_map.height):
                if abs(x - rx) + abs(y - ry) <= self.range_radius:
                    visible.append((x, y))
        return visible

    def _visible_cells_euclidean(self, env_map: GridMap, robot_pos: Position) -> List[Position]:
        rx, ry = robot_pos
        visible: List[Position] = []
        radius_sq = self.range_radius * self.range_radius
        for x in range(env_map.width):
            for y in range(env_map.height):
                dx = x - rx
                dy = y - ry
                if dx * dx + dy * dy <= radius_sq:
                    visible.append((x, y))
        return visible

    def _visible_cells_occluded_manhattan(self, env_map: GridMap, robot_pos: Position) -> List[Position]:
        rx, ry = robot_pos
        visible: List[Position] = []
        for x in range(env_map.width):
            for y in range(env_map.height):
                pos = (x, y)
                if abs(x - rx) + abs(y - ry) > self.range_radius:
                    continue
                if self._line_of_sight_clear(env_map, robot_pos, pos):
                    visible.append(pos)
        return visible

    def _line_of_sight_clear(self, env_map: GridMap, start: Position, end: Position) -> bool:
        """
        Line-of-sight visibility with obstacle blocking.

        The endpoint itself remains visible even if it is an obstacle, but any
        blocking obstacle before the endpoint breaks visibility.
        """
        if start == end:
            return True

        line = self._bresenham_line(start, end)
        for pos in line[1:]:
            if pos == end:
                return True
            if env_map.is_static_blocked(pos) or env_map.is_dynamic_blocked(pos):
                return False
        return True

    def _bresenham_line(self, start: Position, end: Position) -> List[Position]:
        x0, y0 = start
        x1, y1 = end

        points: List[Position] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if (x, y) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points
