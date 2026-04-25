import random
from collections import deque
from typing import List, Tuple

from config.schema import MapConfig
from environment.grid_map import GridMap, Position


class MapGenerator:
    """
    地图生成器 / Map generator

    当前版本采用：
    1. 边界墙体
    2. 内部随机障碍
    3. 连通性约束检测

    Current version uses:
    1. border walls
    2. random internal obstacles
    3. connectivity-preserving checks
    """

    def __init__(self, config: MapConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)

    def generate(self) -> GridMap:
        """
        生成地图 / Generate a map

        采用“增量式障碍放置 + 连通性检测”策略，
        避免生成封闭不可达区域。

        Uses incremental obstacle placement with connectivity checks
        to avoid sealed unreachable regions.
        """
        grid_map = GridMap(self.config.width, self.config.height)
        self._add_border_walls(grid_map)
        self._add_connected_random_obstacles(grid_map)
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

    def _add_connected_random_obstacles(self, grid_map: GridMap) -> None:
        """
        添加满足连通性约束的随机障碍 / Add random obstacles while preserving connectivity

        逻辑：
        - 每次随机选择一个内部格子作为候选障碍
        - 临时放置后检查 free space 是否仍全连通
        - 若仍连通，则保留；否则撤销

        Logic:
        - randomly choose an interior candidate cell
        - tentatively place an obstacle
        - keep it only if the free space remains fully connected
        """
        interior_cell_count = (grid_map.width - 2) * (grid_map.height - 2)
        target_obstacle_count = int(interior_cell_count * self.config.obstacle_density)

        placed_count = 0
        attempts = 0
        max_attempts = max(500, target_obstacle_count * 50)

        while placed_count < target_obstacle_count and attempts < max_attempts:
            attempts += 1

            pos = self._sample_interior_cell(grid_map)

            # 已经是障碍就跳过 / Skip if already occupied
            if pos in grid_map.static_obstacles:
                continue

            # 若放置后仍保持连通，则接受 / Accept only if connectivity is preserved
            if self._can_place_obstacle_without_disconnect(grid_map, pos):
                grid_map.add_static_obstacle(pos)
                placed_count += 1

        # 可选：你也可以在这里打印 placed_count 作为调试信息
        # Optional: print placed_count here for debugging if needed

    def _sample_interior_cell(self, grid_map: GridMap) -> Position:
        """
        随机采样内部格子 / Randomly sample an interior cell
        """
        x = self.random.randint(1, grid_map.width - 2)
        y = self.random.randint(1, grid_map.height - 2)
        return (x, y)

    def _can_place_obstacle_without_disconnect(self, grid_map: GridMap, pos: Position) -> bool:
        """
        检查在某个位置放置障碍后是否仍保持 free space 连通
        Check whether placing an obstacle at 'pos' preserves free-space connectivity
        """
        # 临时放置障碍 / Temporarily place the obstacle
        grid_map.static_obstacles.add(pos)

        is_connected = self._is_free_space_connected(grid_map)

        # 撤销临时放置 / Revert temporary placement
        grid_map.static_obstacles.remove(pos)

        return is_connected

    def _is_free_space_connected(self, grid_map: GridMap) -> bool:
        """
        判断所有 free cells 是否属于同一个连通分量
        Check whether all free cells belong to one connected component
        """
        free_cells = self._get_all_free_cells(grid_map)

        # 没有 free cell 的情况视为不合法
        # If no free cells remain, treat as invalid
        if not free_cells:
            return False

        start = free_cells[0]
        visited = self._bfs_free_space(grid_map, start)

        return len(visited) == len(free_cells)

    def _get_all_free_cells(self, grid_map: GridMap) -> List[Position]:
        """
        获取所有自由格子 / Get all free cells
        """
        free_cells: List[Position] = []

        for x in range(1, grid_map.width - 1):
            for y in range(1, grid_map.height - 1):
                pos = (x, y)
                if pos not in grid_map.static_obstacles:
                    free_cells.append(pos)

        return free_cells

    def _bfs_free_space(self, grid_map: GridMap, start: Position) -> set[Position]:
        """
        在自由空间上做 BFS / Run BFS over free-space cells
        """
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            for nb in grid_map.get_neighbors(current):
                if nb in visited:
                    continue
                if nb in grid_map.static_obstacles:
                    continue
                if not grid_map.in_bounds(nb):
                    continue

                visited.add(nb)
                queue.append(nb)

        return visited

    def sample_robot_starts(self, grid_map: GridMap, n_agents: int) -> List[Position]:
        """
        采样机器人起点 / Sample robot start positions

        由于地图已保证 free space 连通，
        任意采样到的 free cell 理论上都能到达整个自由区域。

        Since free space is guaranteed to be connected,
        any sampled free cell can theoretically reach the entire free region.
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
    def sample_charging_stations(
        self,
        grid_map: GridMap,
        count: int,
        forbidden: List[Position] | None = None,
    ) -> List[Position]:
        """
        采样公共充电站位置 / Sample public charging station positions.

        These are additional charging stations beyond the robots' start cells.
        """
        if count <= 0:
            return []

        forbidden_set = set(forbidden or [])
        stations: List[Position] = []
        tried = 0
        max_tries = 5000

        while len(stations) < count and tried < max_tries:
            tried += 1
            x = self.random.randint(1, grid_map.width - 2)
            y = self.random.randint(1, grid_map.height - 2)
            pos = (x, y)

            if (
                grid_map.is_walkable(pos)
                and pos not in forbidden_set
                and pos not in stations
                and pos not in grid_map.charging_stations
            ):
                stations.append(pos)

        if len(stations) < count:
            raise RuntimeError("Unable to sample enough valid charging station positions.")

        return stations
