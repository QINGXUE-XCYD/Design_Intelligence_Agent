import heapq
from typing import Dict, List, Optional, Tuple

from mapping.occupancy_grid import OccupancyGrid, OccupancyState
from planning.planner_base import PlannerBase, Position


class AStarPlanner(PlannerBase):
    """
    A* 路径规划器 / A* path planner

    第一阶段默认只允许在已知 free 区域规划。
    In phase 1, planning is restricted to known free cells.
    """

    def heuristic(self, a: Position, b: Position) -> float:
        """
        启发函数 / Heuristic function
        使用曼哈顿距离。
        Uses Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start: Position, goal: Position, belief_map: OccupancyGrid) -> List[Position]:
        """
        规划路径 / Plan path

        返回从 start 到 goal 的路径。
        Return a path from start to goal.
        若无法到达，返回空列表。
        Return an empty list if goal is unreachable.
        """
        if start == goal:
            return [start]

        open_heap: List[Tuple[float, Position]] = []
        heapq.heappush(open_heap, (0, start))

        came_from: Dict[Position, Optional[Position]] = {start: None}
        g_score: Dict[Position, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nb in belief_map.get_neighbors(current):
                if nb != goal and belief_map.get_cell(nb) != OccupancyState.FREE:
                    continue

                tentative_g = g_score[current] + 1.0
                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f_score, nb))

        return []

    def _reconstruct_path(
        self,
        came_from: Dict[Position, Optional[Position]],
        current: Position
    ) -> List[Position]:
        """
        回溯构造路径 / Reconstruct path by backtracking
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path