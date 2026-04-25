from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from environment.cell import CellType, CleanState

Position = Tuple[int, int]


class GridMap:
    """
    真实环境地图 / Ground-truth environment map.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self.static_obstacles: Set[Position] = set()
        self.dynamic_obstacles: Set[Position] = set()
        self.cleaned_cells: Set[Position] = set()
        self.cleaned_by: Dict[Position, int] = {}
        self.charging_stations: Set[Position] = set()
        self.home_charging_stations: Set[Position] = set()
        self.public_charging_stations: Set[Position] = set()

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_static_blocked(self, pos: Position) -> bool:
        return pos in self.static_obstacles

    def is_dynamic_blocked(self, pos: Position) -> bool:
        return pos in self.dynamic_obstacles

    def is_walkable(self, pos: Position) -> bool:
        return (
            self.in_bounds(pos)
            and not self.is_static_blocked(pos)
            and not self.is_dynamic_blocked(pos)
        )

    def add_static_obstacle(self, pos: Position) -> None:
        if self.in_bounds(pos):
            self.static_obstacles.add(pos)

    def add_dynamic_obstacle(self, pos: Position) -> None:
        if self.in_bounds(pos) and not self.is_static_blocked(pos):
            self.dynamic_obstacles.add(pos)

    def clear_dynamic_obstacles(self) -> None:
        self.dynamic_obstacles.clear()

    def add_charging_station(self, pos: Position, station_type: str = "public") -> None:
        if self.is_walkable(pos):
            self.charging_stations.add(pos)
            if station_type == "home":
                self.home_charging_stations.add(pos)
            else:
                self.public_charging_stations.add(pos)

    def mark_cleaned(self, pos: Position, robot_id: Optional[int] = None) -> bool:
        """
        标记某个格子已清扫 / Mark a cell as cleaned.

        Returns True only when the cell was newly cleaned by this call.
        """
        if not self.in_bounds(pos) or self.is_static_blocked(pos):
            return False
        was_new = pos not in self.cleaned_cells
        self.cleaned_cells.add(pos)
        if was_new and robot_id is not None:
            self.cleaned_by[pos] = robot_id
        return was_new

    def is_cleaned(self, pos: Position) -> bool:
        return pos in self.cleaned_cells

    def get_cell_type(self, pos: Position) -> CellType:
        if self.is_static_blocked(pos):
            return CellType.STATIC_OBSTACLE
        if self.is_dynamic_blocked(pos):
            return CellType.DYNAMIC_OBSTACLE
        return CellType.FREE

    def get_clean_state(self, pos: Position) -> CleanState:
        return CleanState.CLEANED if self.is_cleaned(pos) else CleanState.DIRTY

    def get_neighbors(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in candidates if self.in_bounds(p)]

    def get_walkable_neighbors(self, pos: Position) -> List[Position]:
        return [p for p in self.get_neighbors(pos) if self.is_walkable(p)]

    def count_cleanable_cells(self) -> int:
        total = 0
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if not self.is_static_blocked(pos):
                    total += 1
        return total

    def get_cleanable_cells(self) -> List[Position]:
        cells: List[Position] = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if not self.is_static_blocked(pos):
                    cells.append(pos)
        return cells
