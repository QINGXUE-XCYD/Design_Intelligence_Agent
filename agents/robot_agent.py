from __future__ import annotations
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
            goal = self.choose_goal()
            self.state.current_goal = goal

            if goal is None:
                self.state.mode = AgentMode.DONE
                self.state.done_reason = "no_frontier_available"
                self.mark_current_cell_cleaned(env_map)
                return

            self.state.current_path = self.plan_path(goal)

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
            else:
                self.state.total_path_length += 1.0
                self.state.position = new_pos
                self.state.trajectory.append(new_pos)  # 记录轨迹 / append trajectory
                self.state.current_path = self.state.current_path[1:]
        else:
            self.state.mode = AgentMode.IDLE
            self.state.idle_steps += 1

        self.state.mode = AgentMode.CLEANING
        self.mark_current_cell_cleaned(env_map)