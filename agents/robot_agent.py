from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from agents.agent_state import AgentMode, AgentState
from control.executor import ActionExecutor
from environment.grid_map import GridMap, Position
from exploration.frontier_detector import FrontierDetector
from mapping.occupancy_grid import OccupancyGrid, OccupancyState
from planning.planner_base import PlannerBase
from sensing.observation import Observation
from sensing.sensor_model import SensorModel


class RobotAgent:
    """
    机器人智能体 / Robot agent

    每个机器人独立维护：
    - 自身状态
    - 内部地图
    - 感知模块
    - 探索模块
    - 路径规划模块
    - 动作执行模块

    Each robot independently maintains:
    - its own runtime state
    - belief map
    - sensing module
    - exploration module
    - planner
    - action execution module
    """

    def __init__(
            self,
            robot_id: int,
            start_pos: Position,
            belief_map: OccupancyGrid,
            sensor_model: SensorModel,
            frontier_detector: FrontierDetector,
            planner: PlannerBase,
            executor: ActionExecutor,
    ) -> None:
        self.state = AgentState(robot_id=robot_id, position=start_pos)
        self.state.trajectory.append(start_pos)  # 记录初始位置 / record initial position
        self.belief_map = belief_map
        self.sensor_model = sensor_model
        self.frontier_detector = frontier_detector
        self.planner = planner
        self.executor = executor
        # 是否已经从探索阶段切换到补扫阶段
        # Whether the robot has switched from exploration to cleanup
        self.in_cleanup_phase = False
        # 补扫阶段开始时对应的轨迹索引与坐标
        # Trajectory split index and position when cleanup starts
        self.cleanup_start_traj_index: int | None = None
        self.cleanup_cleaned_cells: set[Position] = set()

    def _is_reachable_in_truth(self, env_map: GridMap, start: Position, goal: Position) -> bool:
        """
        在真实环境地图上检查可达性 / Check reachability on the ground-truth map
        """
        if start == goal:
            return True
        if not env_map.is_walkable(goal):
            return False

        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            for nb in env_map.get_neighbors(current):
                if nb in visited:
                    continue
                if not env_map.is_walkable(nb):
                    continue

                if nb == goal:
                    return True

                visited.add(nb)
                queue.append(nb)

        return False

    def _is_reachable_in_belief(self, start: Position, goal: Position) -> bool:
        """
        在 belief map 上检查可达性 / Check reachability on the belief map

        这里直接复用 planner：
        - 若 path 长度 >= 2，则视为可达
        - 若 start == goal，也视为可达
        """
        if start == goal:
            return True

        path = self.plan_path(goal)
        return len(path) >= 2

    def _debug_frontier_failure(self, env_map: GridMap, failed_goal: Position) -> None:
        """
        当选中的 frontier 不可规划时，输出调试信息
        Print debug information when the selected frontier cannot be planned to
        """
        start = self.state.position
        frontiers = self.frontier_detector.detect(self.belief_map)

        print("\n[DEBUG] Frontier planning failure detected")
        print(f"[DEBUG] robot_id={self.state.robot_id}")
        print(f"[DEBUG] current_position={start}")
        print(f"[DEBUG] selected_goal={failed_goal}")
        print(f"[DEBUG] frontier_count={len(frontiers)}")

        truth_reachable = self._is_reachable_in_truth(env_map, start, failed_goal)
        belief_reachable = self._is_reachable_in_belief(start, failed_goal)

        print(f"[DEBUG] selected_goal_truth_reachable={truth_reachable}")
        print(f"[DEBUG] selected_goal_belief_reachable={belief_reachable}")

        print("[DEBUG] first 10 frontiers:")
        for i, frontier in enumerate(frontiers[:10]):
            path = self.plan_path(frontier)
            f_truth = self._is_reachable_in_truth(env_map, start, frontier)
            f_belief = len(path) >= 2 or frontier == start
            print(
                f"  frontier[{i}]={frontier}, "
                f"truth_reachable={f_truth}, "
                f"belief_reachable={f_belief}, "
                f"path_len={len(path)}"
            )

    def _manhattan_distance(self, a: Position, b: Position) -> int:
        """
        计算曼哈顿距离 / Compute Manhattan distance
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def choose_reachable_goal(self) -> tuple[Optional[Position], List[Position], str]:
        """
        选择一个可达的 frontier 目标 / Choose a reachable frontier goal

        返回：
        - goal: 选中的目标
        - path: 到目标的路径
        - reason:
            "reachable_frontier_found"
            "no_frontier_available"
            "no_reachable_frontier"
        """
        frontiers = self.frontier_detector.detect(self.belief_map)

        if not frontiers:
            return None, [], "no_frontier_available"

        # 先按距离排序 / Sort by distance first
        frontiers = sorted(
            frontiers,
            key=lambda p: self._manhattan_distance(self.state.position, p)
        )

        # 逐个测试可达性 / Test reachability one by one
        for frontier in frontiers:
            path = self.plan_path(frontier)
            if frontier == self.state.position or len(path) >= 2:
                return frontier, path, "reachable_frontier_found"

        return None, [], "no_reachable_frontier"

    def _belief_cell_is_cleaned(self, pos: Position) -> bool:
        """
        判断 belief map 中某个格子是否已被认为清扫过
        Check whether a cell is believed to be cleaned
        """
        return self.belief_map.is_cleaned(pos)

    def _get_dirty_candidates(self) -> List[Position]:
        """
        获取 belief map 中所有“已知为空闲但尚未清扫”的候选格子
        Get all candidate cells that are known FREE but not yet cleaned
        """
        return self.belief_map.get_dirty_free_cells()

    def choose_reachable_dirty_goal(self) -> tuple[Optional[Position], List[Position], str]:
        """
        选择一个可达的补扫目标 / Choose a reachable cleanup target

        返回：
        - goal: 选中的脏格子
        - path: 到目标的路径
        - reason:
            "reachable_dirty_found"
            "no_dirty_remaining"
            "no_reachable_dirty"
        """
        dirty_cells = self._get_dirty_candidates()

        if not dirty_cells:
            return None, [], "no_dirty_remaining"

        dirty_cells = sorted(
            dirty_cells,
            key=lambda p: self._manhattan_distance(self.state.position, p)
        )

        for cell in dirty_cells:
            path = self.plan_path(cell)
            if cell == self.state.position or len(path) >= 2:
                return cell, path, "reachable_dirty_found"

        return None, [], "no_reachable_dirty"

    def perceive(self, env_map: GridMap) -> Observation:
        """
        感知环境 / Perceive the environment
        """
        return self.sensor_model.sense(env_map, self.state.position)

    def update_belief(self, observation: Observation) -> None:
        """
        根据观测更新内部地图 / Update belief map using an observation
        """
        for pos in observation.free_cells:
            self.belief_map.update_cell(pos, OccupancyState.FREE)

        for pos in observation.occupied_cells:
            self.belief_map.update_cell(pos, OccupancyState.OCCUPIED)

        for pos in observation.dynamic_cells:
            self.belief_map.update_cell(pos, OccupancyState.OCCUPIED)

    def choose_goal(self) -> Optional[Position]:
        """
        选择目标 / Choose a target goal

        第一阶段：选择第一个 frontier 作为目标。
        Phase 1: choose the first detected frontier as the goal.
        """
        frontiers = self.frontier_detector.detect(self.belief_map)
        if not frontiers:
            return None
        return frontiers[0]

    def plan_path(self, goal: Position) -> List[Position]:
        """
        规划到目标的路径 / Plan a path to the selected goal
        """
        return self.planner.plan(self.state.position, goal, self.belief_map)

    def mark_current_cell_cleaned(self, env_map: GridMap) -> None:
        """
        清扫当前位置 / Clean the current cell
        """
        pos = self.state.position
        if not env_map.is_cleaned(pos):
            env_map.mark_cleaned(pos)
            self.belief_map.mark_cleaned(pos)
            self.state.cleaned_cells += 1
            # 记录 cleanup 阶段首次清扫到的格子
            # Record cells first cleaned during cleanup phase
            if getattr(self, 'in_cleanup_phase', False):
                self.cleanup_cleaned_cells.add(pos)

    def step(self, env_map: GridMap) -> None:
        """
        单步决策与执行 / One decision-execution cycle

        Phase 1 flow:
        1. sense
        2. update belief
        3. choose goal if needed
        4. plan path if needed
        5. move one step
        6. clean current cell
        """
        # 如果已经 DONE，则不再执行 / If already done, do nothing
        if self.state.mode == AgentMode.DONE:
            return
        self.state.steps_taken += 1

        observation = self.perceive(env_map)
        self.update_belief(observation)

        self.belief_map.update_cell(self.state.position, OccupancyState.FREE)

        if (
                self.state.current_goal is None
                or self.state.position == self.state.current_goal
                or not self.state.current_path
        ):
            # 先清空旧目标，准备重新选
            # Clear the old goal before selecting a new one
            self.state.current_goal = None
            self.state.current_path = []

            # =========================
            # 阶段 1：探索 / Exploration
            # =========================
            if not self.in_cleanup_phase:
                goal, path, reason = self.choose_reachable_goal()

                if goal is not None:
                    self.state.current_goal = goal
                    self.state.current_path = path

                    # frontier 调试逻辑只保留在探索阶段
                    # Keep frontier debug logic only in exploration phase
                    if goal != self.state.position and len(self.state.current_path) < 2:
                        self._debug_frontier_failure(env_map, goal)
                else:
                    # frontier 用完了，切换到补扫阶段
                    # No reachable frontier left, switch to cleanup phase
                    self.in_cleanup_phase = True
                    # 记录 explore -> cleanup 的切换点
                    # Record the switch point from exploration to cleanup
                    if self.cleanup_start_traj_index is None:
                        self.cleanup_start_traj_index = max(0, len(self.state.trajectory) - 1)

            # =========================
            # 阶段 2：补扫 / Cleanup
            # =========================
            if self.in_cleanup_phase and self.state.current_goal is None:
                goal, path, reason = self.choose_reachable_dirty_goal()

                if goal is not None:
                    self.state.current_goal = goal
                    self.state.current_path = path
                else:
                    # 没有剩余可补扫格子，真正完成任务
                    # No remaining reachable dirty cells, coverage completed
                    self.state.mode = AgentMode.DONE
                    self.state.done_reason = "coverage_completed"
                    self.mark_current_cell_cleaned(env_map)
                    return

        if len(self.state.current_path) >= 2:
            self.state.mode = AgentMode.MOVING
            action = self.executor.next_action_from_path(
                self.state.current_path,
                self.state.position
            )
            new_pos = self.executor.execute_move(
                self.state.position,
                action,
                env_map
            )

            if new_pos == self.state.position:
                self.state.mode = AgentMode.BLOCKED
                self.state.idle_steps += 1
                # 被阻塞后清空当前目标，下一步触发重选/重规划
                # Clear current goal when blocked so next step can reselect/replan
                self.state.current_goal = None
                self.state.current_path = []
            else:
                self.state.total_path_length += 1.0
                self.state.position = new_pos
                self.state.trajectory.append(new_pos)  # 记录轨迹 / append trajectory
                self.state.current_path = self.state.current_path[1:]
        else:
            self.state.mode = AgentMode.IDLE
            self.state.idle_steps += 1

        self.mark_current_cell_cleaned(env_map)
