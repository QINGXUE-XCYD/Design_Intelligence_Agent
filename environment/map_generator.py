import random
from typing import List, Tuple

from config.schema import MapConfig
from environment.grid_map import GridMap, Position


class MapGenerator:
    """
    地图生成器 / Map generator

    用于生成实验场景中的二维栅格地图。
    Used to generate 2D grid maps for experiments.
    """

    def __init__(self, config: MapConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)

    def generate(self) -> GridMap:
        """
        生成地图 / Generate a map

        第一阶段先实现简单随机障碍环境。
        In phase 1, we start with a simple random obstacle layout.
        """
        grid_map = GridMap(self.config.width, self.config.height)
        self._add_border_walls(grid_map)
        self._add_random_obstacles(grid_map)
        return grid_map

    def _add_border_walls(self, grid_map: GridMap) -> None:
        """
        添加边界墙体 / Add border walls
        """
        for x in range(grid_map.width):
            grid_map.add_static_obstacle((x, 0))
            grid_map.add_static_obstacle((x, grid_map.height - 1))

        for y in range(grid_map.height):
            grid_map.add_static_obstacle((0, y))
            grid_map.add_static_obstacle((grid_map.width - 1, y))

    def _add_random_obstacles(self, grid_map: GridMap) -> None:
        """
        添加随机静态障碍 / Add random static obstacles
        """
        total_cells = grid_map.width * grid_map.height
        obstacle_count = int(total_cells * self.config.obstacle_density)

        for _ in range(obstacle_count):
            x = self.random.randint(1, grid_map.width - 2)
            y = self.random.randint(1, grid_map.height - 2)
            grid_map.add_static_obstacle((x, y))

    def sample_robot_starts(self, grid_map: GridMap, n_agents: int) -> List[Position]:
        """
        采样机器人起点 / Sample robot start positions

        要求起点可通行且互不重复。
        Start positions must be walkable and unique.
        """
        starts = []
        tried = 0
        max_tries = 5000

        while len(starts) < n_agents and tried < max_tries:
            tried += 1
            x = self.random.randint(1, grid_map.width - 2)
            y = self.random.randint(1, grid_map.height - 2)
            pos = (x, y)

            if grid_map.is_walkable(pos) and pos not in starts:
                starts.append(pos)

        if len(starts) < n_agents:
            raise RuntimeError("Unable to sample enough valid robot start positions.")

        return starts