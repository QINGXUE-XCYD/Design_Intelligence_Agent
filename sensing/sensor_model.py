from typing import List, Tuple

from environment.cell import CellType
from environment.grid_map import GridMap, Position
from sensing.observation import Observation


class SensorModel:
    """
    传感器模型 / Sensor model

    第一阶段使用简单的固定半径感知。
    Phase 1 uses a simple fixed-range sensing model.
    """

    def __init__(self, range_radius: int) -> None:
        self.range_radius = range_radius

    def sense(self, env_map: GridMap, robot_pos: Position) -> Observation:
        """
        获取机器人当前位置附近的局部观测 / Sense a local observation around robot position
        """
        visible = self.get_visible_cells(env_map, robot_pos)

        obs = Observation()
        obs.visible_cells = visible

        for pos in visible:
            cell_type = env_map.get_cell_type(pos)
            if cell_type == CellType.FREE:
                obs.free_cells.append(pos)
            elif cell_type == CellType.STATIC_OBSTACLE:
                obs.occupied_cells.append(pos)
            elif cell_type == CellType.DYNAMIC_OBSTACLE:
                obs.dynamic_cells.append(pos)

        return obs

    def get_visible_cells(self, env_map: GridMap, robot_pos: Position) -> List[Position]:
        """
        获取可见区域 / Get visible cells

        第一阶段用曼哈顿距离范围近似可见域。
        In phase 1, visible area is approximated using a Manhattan-distance range.
        """
        rx, ry = robot_pos
        visible = []

        for x in range(env_map.width):
            for y in range(env_map.height):
                if abs(x - rx) + abs(y - ry) <= self.range_radius:
                    visible.append((x, y))

        return visible