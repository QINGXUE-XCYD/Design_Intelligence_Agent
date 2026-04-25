from __future__ import annotations
from enum import Enum
from typing import Dict, List, Tuple

Position = Tuple[int, int]


class OccupancyState(Enum):
    """
    机器人认知地图中的占据状态 / Occupancy state in the robot belief map.
    """
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


class OccupancyGrid:
    """
    机器人内部地图 / Robot belief map.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.occupancy = [
            [OccupancyState.UNKNOWN for _ in range(height)]
            for _ in range(width)
        ]
        self.cleaned = [
            [False for _ in range(height)]
            for _ in range(width)
        ]

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def update_cell(self, pos: Position, state: OccupancyState) -> None:
        if self.in_bounds(pos):
            x, y = pos
            self.occupancy[x][y] = state

    def get_cell(self, pos: Position) -> OccupancyState:
        x, y = pos
        return self.occupancy[x][y]

    def mark_cleaned(self, pos: Position) -> None:
        if self.in_bounds(pos):
            x, y = pos
            self.cleaned[x][y] = True

    def is_cleaned(self, pos: Position) -> bool:
        x, y = pos
        return self.cleaned[x][y]

    def is_known(self, pos: Position) -> bool:
        return self.get_cell(pos) != OccupancyState.UNKNOWN

    def get_neighbors(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in candidates if self.in_bounds(p)]

    def known_free_cells(self) -> List[Position]:
        cells: List[Position] = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if self.get_cell(pos) == OccupancyState.FREE:
                    cells.append(pos)
        return cells

    def known_free_uncleaned_cells(self) -> List[Position]:
        return [p for p in self.known_free_cells() if not self.is_cleaned(p)]

    def count_states(self) -> Dict[str, int]:
        counts = {"unknown": 0, "free": 0, "occupied": 0, "cleaned": 0}
        for x in range(self.width):
            for y in range(self.height):
                state = self.occupancy[x][y]
                if state == OccupancyState.UNKNOWN:
                    counts["unknown"] += 1
                elif state == OccupancyState.FREE:
                    counts["free"] += 1
                elif state == OccupancyState.OCCUPIED:
                    counts["occupied"] += 1
                if self.cleaned[x][y]:
                    counts["cleaned"] += 1
        return counts

    def merge(self, other: "OccupancyGrid") -> None:
        """
        地图融合接口 / Map fusion interface.

        Occupied observations are conservative and override free observations;
        cleaned flags are merged by logical OR.
        """
        for x in range(self.width):
            for y in range(self.height):
                other_state = other.occupancy[x][y]
                current_state = self.occupancy[x][y]

                if other_state == OccupancyState.OCCUPIED:
                    self.occupancy[x][y] = OccupancyState.OCCUPIED
                elif other_state == OccupancyState.FREE and current_state == OccupancyState.UNKNOWN:
                    self.occupancy[x][y] = OccupancyState.FREE

                if other.cleaned[x][y]:
                    self.cleaned[x][y] = True
