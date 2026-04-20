from __future__ import annotations
from typing import List, Tuple, Set

from environment.cell import CellType, CleanState

Position = Tuple[int, int]


class GridMap:
    """
    真实环境地图 / Ground-truth environment map

    该类维护仿真中的真实世界状态，包括：
    - 静态障碍
    - 动态障碍（第一阶段先预留接口）
    - 可清扫区域及其清扫状态

    This class stores the ground-truth world state in simulation, including:
    - static obstacles
    - dynamic obstacles (reserved interface in phase 1)
    - cleanable cells and their cleaning states
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self.static_obstacles: Set[Position] = set()
        self.dynamic_obstacles: Set[Position] = set()
        self.cleaned_cells: Set[Position] = set()

    def in_bounds(self, pos: Position) -> bool:
        """
        判断坐标是否在地图范围内 / Check whether a position is inside map bounds
        """
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_static_blocked(self, pos: Position) -> bool:
        """
        判断是否为静态障碍 / Check whether the position is blocked by a static obstacle
        """
        return pos in self.static_obstacles

    def is_dynamic_blocked(self, pos: Position) -> bool:
        """
        判断是否为动态障碍 / Check whether the position is blocked by a dynamic obstacle
        """
        return pos in self.dynamic_obstacles

    def is_walkable(self, pos: Position) -> bool:
        """
        判断该位置是否可通行 / Check whether a cell is walkable

        第一阶段定义：
        - 在边界内
        - 不是静态障碍
        - 不是动态障碍

        Phase-1 definition:
        - inside bounds
        - not a static obstacle
        - not a dynamic obstacle
        """
        return (
            self.in_bounds(pos)
            and not self.is_static_blocked(pos)
            and not self.is_dynamic_blocked(pos)
        )

    def add_static_obstacle(self, pos: Position) -> None:
        """
        添加静态障碍 / Add a static obstacle
        """
        if self.in_bounds(pos):
            self.static_obstacles.add(pos)

    def add_dynamic_obstacle(self, pos: Position) -> None:
        """
        添加动态障碍 / Add a dynamic obstacle
        """
        if self.in_bounds(pos):
            self.dynamic_obstacles.add(pos)

    def clear_dynamic_obstacles(self) -> None:
        """
        清空动态障碍 / Clear all dynamic obstacles
        """
        self.dynamic_obstacles.clear()

    def mark_cleaned(self, pos: Position) -> None:
        """
        标记某个格子已清扫 / Mark a cell as cleaned
        """
        if self.in_bounds(pos) and not self.is_static_blocked(pos):
            self.cleaned_cells.add(pos)

    def is_cleaned(self, pos: Position) -> bool:
        """
        判断某个格子是否已清扫 / Check whether a cell has been cleaned
        """
        return pos in self.cleaned_cells

    def get_cell_type(self, pos: Position) -> CellType:
        """
        返回真实环境中的格子类型 / Return the ground-truth cell type
        """
        if self.is_static_blocked(pos):
            return CellType.STATIC_OBSTACLE
        if self.is_dynamic_blocked(pos):
            return CellType.DYNAMIC_OBSTACLE
        return CellType.FREE

    def get_clean_state(self, pos: Position) -> CleanState:
        """
        返回格子的清扫状态 / Return the cleaning state of a cell
        """
        return CleanState.CLEANED if self.is_cleaned(pos) else CleanState.DIRTY

    def get_neighbors(self, pos: Position) -> List[Position]:
        """
        获取四邻域 / Get 4-connected neighbors
        """
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in candidates if self.in_bounds(p)]

    def get_walkable_neighbors(self, pos: Position) -> List[Position]:
        """
        获取可通行邻居 / Get walkable neighboring cells
        """
        return [p for p in self.get_neighbors(pos) if self.is_walkable(p)]

    def count_cleanable_cells(self) -> int:
        """
        统计理论上可清扫的总格子数 / Count total cleanable cells
        """
        total = 0
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if not self.is_static_blocked(pos):
                    total += 1
        return total