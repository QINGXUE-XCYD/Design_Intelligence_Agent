from __future__ import annotations

from collections import deque
from typing import List, Optional

from agents.agent_state import AgentMode, AgentRole, AgentState
from control.executor import ActionExecutor
from coordination.reservation_table import GoalReservationTable
from environment.grid_map import GridMap, Position
from exploration.frontier_detector import FrontierDetector
from mapping.occupancy_grid import OccupancyGrid, OccupancyState
from planning.planner_base import PlannerBase
from sensing.observation import Observation
from sensing.sensor_model import SensorModel


class RobotAgent:
    """
    机器人智能体 / Robot agent

    当前版本支持固定角色协同：
    - scout: 只负责探索开图
    - cleaner: 只负责清扫
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
            role: AgentRole,
            active: bool = True,
            reservation_table: GoalReservationTable | None = None,
    ) -> None:
        initial_mode = AgentMode.EXPLORING if role == AgentRole.SCOUT and active else AgentMode.WAITING
        self.state = AgentState(
            robot_id=robot_id,
            position=start_pos,
            role=role,
            mode=initial_mode,
            activation_step=0 if active else None,
        )
        self.state.trajectory.append(start_pos)

        self.belief_map = belief_map
        self.sensor_model = sensor_model
        self.frontier_detector = frontier_detector
        self.planner = planner
        self.executor = executor
        self.reservation_table = reservation_table
        self.active = active

        self.in_cleanup_phase = role == AgentRole.CLEANER and active
        self.cleanup_start_traj_index: int | None = 0 if self.in_cleanup_phase else None
        self.cleanup_cleaned_cells: set[Position] = set()

        # 机器人知道自身起始位置，写入共享/本地地图。
        self.belief_map.update_cell(start_pos, OccupancyState.FREE)

    def activate(self, step_idx: int) -> None:
        """
        激活等待中的机器人 / Activate a waiting robot
        """
        if self.active:
            return

        self.active = True
        self.state.activation_step = step_idx
        self.state.mode = AgentMode.IDLE

        if self.state.role == AgentRole.CLEANER:
            self.in_cleanup_phase = True
            self.cleanup_start_traj_index = max(0, len(self.state.trajectory) - 1)

    def _reset_current_goal(self) -> None:
        """
        清空当前目标并释放预留 / Clear current goal and release reservation
        """
        if self.state.role == AgentRole.CLEANER and self.reservation_table is not None:
            self.reservation_table.release(self.state.robot_id)

        self.state.current_goal = None
        self.state.current_path = []

    def _assign_goal(self, goal: Position, path: List[Position]) -> None:
        """
        设置当前目标 / Assign current goal
        """
        self.state.current_goal = goal
        self.state.current_path = path

        if self.state.role == AgentRole.CLEANER and self.reservation_table is not None:
            self.reservation_table.reserve(self.state.robot_id, goal)

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
        """
        frontiers = self.frontier_detector.detect(self.belief_map)

        if not frontiers:
            return None, [], "no_frontier_available"

        frontiers = sorted(
            frontiers,
            key=lambda p: self._manhattan_distance(self.state.position, p)
        )

        for frontier in frontiers:
            path = self.plan_path(frontier)
            if frontier == self.state.position or len(path) >= 2:
                return frontier, path, "reachable_frontier_found"

        return None, [], "no_reachable_frontier"

    def _get_dirty_candidates(self) -> List[Position]:
        """
        获取 belief map 中所有已知但尚未清扫的自由格子
        Get all dirty free cells from the current belief map
        """
        return self.belief_map.get_dirty_free_cells()

    def choose_reachable_dirty_goal(self) -> tuple[Optional[Position], List[Position], str]:
        """
        选择一个可达的补扫目标 / Choose a reachable cleanup target
        """
        dirty_cells = self._get_dirty_candidates()

        if not dirty_cells:
            return None, [], "no_dirty_remaining"

        available_cells = [
            cell
            for cell in dirty_cells
            if self.reservation_table is None
            or not self.reservation_table.is_reserved_by_other(cell, self.state.robot_id)
        ]

        if not available_cells:
            return None, [], "all_dirty_reserved"

        available_cells = sorted(
            available_cells,
            key=lambda p: self._manhattan_distance(self.state.position, p)
        )

        for cell in available_cells:
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

    def plan_path(self, goal: Position) -> List[Position]:
        """
        规划到目标的路径 / Plan a path to the selected goal
        """
        return self.planner.plan(self.state.position, goal, self.belief_map)

    def mark_current_cell_cleaned(self, env_map: GridMap) -> None:
        """
        清扫当前位置 / Clean the current cell
        """
        if self.state.role != AgentRole.CLEANER:
            return

        pos = self.state.position
        if not env_map.is_cleaned(pos):
            env_map.mark_cleaned(pos)
            self.belief_map.mark_cleaned(pos)
            self.state.cleaned_cells += 1
            if self.in_cleanup_phase:
                self.cleanup_cleaned_cells.add(pos)

    def _execute_move_step(self, env_map: GridMap, allow_cleaning: bool) -> None:
        """
        执行一步移动 / Execute one movement step
        """
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
                self._reset_current_goal()
                return

            self.state.total_path_length += 1.0
            self.state.position = new_pos
            self.state.trajectory.append(new_pos)
            self.state.current_path = self.state.current_path[1:]

            if allow_cleaning:
                self.mark_current_cell_cleaned(env_map)
                if self.state.current_goal == self.state.position and self.belief_map.is_cleaned(self.state.position):
                    self._reset_current_goal()
            return

        if allow_cleaning and not env_map.is_cleaned(self.state.position):
            self.state.mode = AgentMode.CLEANING
            self.mark_current_cell_cleaned(env_map)
            if self.state.current_goal == self.state.position:
                self._reset_current_goal()
            return

        self.state.mode = AgentMode.IDLE
        self.state.idle_steps += 1

    def _step_as_scout(self, env_map: GridMap) -> None:
        """
        侦察机器人单步逻辑 / One step for the scout robot
        """
        if (
                self.state.current_goal is None
                or self.state.position == self.state.current_goal
                or not self.state.current_path
        ):
            self._reset_current_goal()
            goal, path, reason = self.choose_reachable_goal()

            if goal is None:
                self.state.mode = AgentMode.DONE
                self.state.done_reason = (
                    "scouting_completed"
                    if reason == "no_frontier_available"
                    else reason
                )
                return

            self._assign_goal(goal, path)

            if goal != self.state.position and len(self.state.current_path) < 2:
                self._debug_frontier_failure(env_map, goal)

        self._execute_move_step(env_map, allow_cleaning=False)
        if self.state.mode not in (AgentMode.MOVING, AgentMode.BLOCKED, AgentMode.DONE):
            self.state.mode = AgentMode.EXPLORING

    def _step_as_cleaner(self, env_map: GridMap) -> None:
        """
        清扫机器人单步逻辑 / One step for a cleaner robot
        """
        if self.state.current_goal is not None and self.belief_map.is_cleaned(self.state.current_goal):
            self._reset_current_goal()

        if (
                self.state.current_goal is None
                or self.state.position == self.state.current_goal
                or not self.state.current_path
        ):
            self._reset_current_goal()
            goal, path, reason = self.choose_reachable_dirty_goal()

            if goal is None:
                if reason == "no_dirty_remaining":
                    self.state.mode = AgentMode.DONE
                    self.state.done_reason = "coverage_completed"
                elif reason == "all_dirty_reserved":
                    self.state.mode = AgentMode.WAITING
                    self.state.idle_steps += 1
                else:
                    self.state.mode = AgentMode.BLOCKED
                    self.state.idle_steps += 1
                    self.state.done_reason = reason
                return

            self._assign_goal(goal, path)

            if goal == self.state.position:
                self.state.mode = AgentMode.CLEANING
                self.mark_current_cell_cleaned(env_map)
                self._reset_current_goal()
                return

        self._execute_move_step(env_map, allow_cleaning=True)

    def step(self, env_map: GridMap) -> None:
        """
        单步决策与执行 / One decision-execution cycle
        """
        if self.state.mode == AgentMode.DONE:
            return

        if not self.active:
            self.state.mode = AgentMode.WAITING
            return

        self.state.steps_taken += 1

        observation = self.perceive(env_map)
        self.update_belief(observation)
        self.belief_map.update_cell(self.state.position, OccupancyState.FREE)

        if self.state.role == AgentRole.SCOUT:
            self._step_as_scout(env_map)
        else:
            self._step_as_cleaner(env_map)
