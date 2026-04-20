from __future__ import annotations
from enum import Enum
from typing import List, Tuple

Position = Tuple[int, int]


class OccupancyState(Enum):
    """
    机器人认知地图中的占据状态 / Occupancy state in the robot belief map
    """
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


class OccupancyGrid:
    """
    机器人内部地图 / Robot belief map

    维护机器人对环境的认知，包括：
    - unknown / free / occupied
    - cleaned state

    Stores the robot's internal belief of the environment, including:
    - unknown / free / occupied
    - cleaned state
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
        """
        判断是否在地图范围内 / Check whether a position is in bounds
        """
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def update_cell(self, pos: Position, state: OccupancyState) -> None:
        """
        更新某个格子的占据状态 / Update occupancy state of a cell
        """
        if self.in_bounds(pos):
            x, y = pos
            self.occupancy[x][y] = state

    def get_cell(self, pos: Position) -> OccupancyState:
        """
        获取某个格子的占据状态 / Get occupancy state of a cell
        """
        x, y = pos
        return self.occupancy[x][y]

    def mark_cleaned(self, pos: Position) -> None:
        """
        在认知地图中标记为已清扫 / Mark a cell as cleaned in belief map
        """
        if self.in_bounds(pos):
            x, y = pos
            self.cleaned[x][y] = True

    def is_cleaned(self, pos: Position) -> bool:
        """
        判断该格子在认知地图中是否已清扫 / Check cleaned state in belief map
        """
        x, y = pos
        return self.cleaned[x][y]

    def is_known(self, pos: Position) -> bool:
        """
        判断该格子是否已被观测 / Check whether a cell has been observed
        """
        return self.get_cell(pos) != OccupancyState.UNKNOWN

    def get_neighbors(self, pos: Position) -> List[Position]:
        """
        获取四邻域 / Get 4-connected neighbors
        """
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in candidates if self.in_bounds(p)]

    def merge(self, other: "OccupancyGrid") -> None:
        """
        地图融合接口 / Map fusion interface

        第一阶段先做最简单的覆盖式融合：
        如果 other 中某格不是 UNKNOWN，则写入当前地图。

        Phase 1 uses a simple overwrite-style fusion:
        if a cell in 'other' is not UNKNOWN, copy it into current map.
        """
        for x in range(self.width):
            for y in range(self.height):
                other_state = other.occupancy[x][y]
                if other_state != OccupancyState.UNKNOWN:
                    self.occupancy[x][y] = other_state

                if other.cleaned[x][y]:
                    self.cleaned[x][y] = True