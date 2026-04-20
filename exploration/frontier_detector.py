from typing import List, Tuple

from mapping.occupancy_grid import OccupancyGrid, OccupancyState

Position = Tuple[int, int]


class FrontierDetector:
    """
    frontier 检测器 / Frontier detector

    frontier 定义：
    一个已知 free 格子，且其邻居中至少有一个 unknown 格子。

    Frontier definition:
    A known free cell that has at least one unknown neighbor.
    """

    def detect(self, belief_map: OccupancyGrid) -> List[Position]:
        """
        检测所有 frontier 格子 / Detect all frontier cells
        """
        frontiers: List[Position] = []

        for x in range(belief_map.width):
            for y in range(belief_map.height):
                pos = (x, y)
                if self._is_frontier(pos, belief_map):
                    frontiers.append(pos)

        return frontiers

    def _is_frontier(self, pos: Position, belief_map: OccupancyGrid) -> bool:
        """
        判断某格子是否为 frontier / Check whether a cell is a frontier
        """
        if belief_map.get_cell(pos) != OccupancyState.FREE:
            return False

        for nb in belief_map.get_neighbors(pos):
            if belief_map.get_cell(nb) == OccupancyState.UNKNOWN:
                return True

        return False