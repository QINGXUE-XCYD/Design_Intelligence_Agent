from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

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
    机器人智能体 / Robot agent.

    Supports:
    1. frontier-based exploration;
    2. known-free uncleaned-cell clean-up;
    3. optional battery-aware planning and charging.

    Charging design:
    - each robot knows its own home charger at the start;
    - public chargers are discovered only when they are inside sensor range;
    - shared-map strategies can share discovered charger locations;
    - battery feasibility uses known paths on the belief map, not only Manhattan
      estimates, so robots are less likely to run out of battery in unknown maps.
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
        enable_battery: bool = False,
        battery_capacity: float = 250.0,
        low_battery_threshold: float = 45.0,
        recharge_rate: float = 25.0,
        battery_safety_margin: float = 15.0,
        move_energy_cost: float = 1.0,
        sensing_energy_cost: float = 0.05,
        cleaning_energy_cost: float = 0.15,
        communication_energy_cost: float = 0.05,
    ) -> None:
        self.state = AgentState(robot_id=robot_id, position=start_pos)
        self.state.trajectory.append(start_pos)
        self.state.charging_station = start_pos
        self.state.assigned_charging_station = start_pos
        self.state.discovered_charging_stations.add(start_pos)
        self.state.battery_level = battery_capacity if enable_battery else None

        self.belief_map = belief_map
        self.sensor_model = sensor_model
        self.frontier_detector = frontier_detector
        self.planner = planner
        self.executor = executor

        self.enable_battery = enable_battery
        self.battery_capacity = battery_capacity
        self.low_battery_threshold = low_battery_threshold
        self.recharge_rate = recharge_rate
        self.battery_safety_margin = battery_safety_margin
        self.move_energy_cost = move_energy_cost
        self.sensing_energy_cost = sensing_energy_cost
        self.cleaning_energy_cost = cleaning_energy_cost
        self.communication_energy_cost = communication_energy_cost

        # The home charger is always known and free in the robot's local map.
        self.belief_map.update_cell(start_pos, OccupancyState.FREE)

    def _is_reachable_in_truth(self, env_map: GridMap, start: Position, goal: Position) -> bool:
        if start == goal:
            return True
        if not env_map.is_walkable(goal):
            return False
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            for nb in env_map.get_neighbors(current):
                if nb in visited or not env_map.is_walkable(nb):
                    continue
                if nb == goal:
                    return True
                visited.add(nb)
                queue.append(nb)
        return False

    def _debug_planning_failure(self, env_map: GridMap, failed_goal: Position) -> None:
        start = self.state.position
        print("\n[DEBUG] Planning failure detected")
        print(f"[DEBUG] robot_id={self.state.robot_id}")
        print(f"[DEBUG] current_position={start}")
        print(f"[DEBUG] selected_goal={failed_goal}")
        print(f"[DEBUG] selected_goal_truth_reachable={self._is_reachable_in_truth(env_map, start, failed_goal)}")

    def _manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _path_move_count(self, path: List[Position]) -> int:
        return max(0, len(path) - 1)

    def _travel_energy_for_moves(self, moves: int, include_cleaning: bool = False) -> float:
        # A move requires both movement and at least one step of sensing/control.
        energy = moves * (self.move_energy_cost + self.sensing_energy_cost)
        if include_cleaning:
            energy += moves * self.cleaning_energy_cost
        return energy

    def _battery_percent(self) -> float:
        if not self.enable_battery or self.state.battery_level is None or self.battery_capacity <= 0:
            return 1.0
        return self.state.battery_level / self.battery_capacity

    def _consume_energy(self, amount: float) -> None:
        if not self.enable_battery or self.state.battery_level is None or amount <= 0:
            return
        before = self.state.battery_level
        self.state.battery_level = max(0.0, self.state.battery_level - amount)
        self.state.energy_used += before - self.state.battery_level

    def consume_communication_energy(self) -> None:
        self._consume_energy(self.communication_energy_cost)

    def _battery_is_low(self) -> bool:
        return (
            self.enable_battery
            and self.state.battery_level is not None
            and self.state.battery_level <= self.low_battery_threshold
        )

    def merge_discovered_chargers(self, chargers: Set[Position]) -> None:
        """Share discovered charging-station locations with this robot."""
        for charger in chargers:
            self.state.discovered_charging_stations.add(charger)
            self.belief_map.update_cell(charger, OccupancyState.FREE)

    def _known_chargers(self) -> List[Position]:
        chargers = sorted(
            self.state.discovered_charging_stations,
            key=lambda p: self._manhattan_distance(self.state.position, p),
        )
        for charger in chargers:
            self.belief_map.update_cell(charger, OccupancyState.FREE)
        return chargers

    def _plan_between_known_cells(self, start: Position, goal: Position) -> List[Position]:
        if start == goal:
            return [start]
        return self.planner.plan(start, goal, self.belief_map)

    def _nearest_reachable_charger_path(
        self,
        start: Optional[Position] = None,
    ) -> tuple[Optional[Position], List[Position]]:
        start = start or self.state.position
        best_station: Optional[Position] = None
        best_path: List[Position] = []

        for charger in self._known_chargers():
            path = self._plan_between_known_cells(start, charger)
            if path and (best_station is None or len(path) < len(best_path)):
                best_station = charger
                best_path = path

        return best_station, best_path

    def _min_required_energy_to_reach_charger(self) -> Optional[float]:
        charger, charger_path = self._nearest_reachable_charger_path(self.state.position)
        if charger is None or not charger_path:
            return None
        moves = self._path_move_count(charger_path)
        return self._travel_energy_for_moves(moves, include_cleaning=False) + self.battery_safety_margin

    def _should_return_to_charge_now(self) -> bool:
        if not self.enable_battery or self.state.battery_level is None:
            return False
        if self._battery_is_low():
            return True
        required = self._min_required_energy_to_reach_charger()
        if required is None:
            return False
        return self.state.battery_level <= required

    def _recharge_if_possible(self, can_charge: bool = True) -> bool:
        """
        Try to recharge when the robot is on a charging station.

        Returns True when the robot spent this step charging or waiting for a
        charging slot, meaning it should not continue to choose a cleaning goal.
        """
        if not self.enable_battery or self.state.battery_level is None:
            return False

        if self.state.battery_level >= self.battery_capacity:
            # Fully charged and ready to work.
            if self.state.target_type == "charger" or self.state.mode in (AgentMode.CHARGING, AgentMode.WAITING_FOR_CHARGE):
                self.state.current_goal = None
                self.state.current_path = []
                self.state.target_type = None
                self.state.mode = AgentMode.IDLE
            return False

        should_charge_now = (
            self.state.target_type == "charger"
            or self.state.mode in (AgentMode.CHARGING, AgentMode.WAITING_FOR_CHARGE)
            or self._battery_is_low()
        )
        if not should_charge_now:
            # Do not stop to top up after tiny communication/sensing costs.
            return False

        if not can_charge:
            self.state.mode = AgentMode.WAITING_FOR_CHARGE
            self.state.idle_steps += 1
            self.state.charge_wait_steps += 1
            return True

        if self.state.mode != AgentMode.CHARGING:
            self.state.charging_events += 1

        self.state.mode = AgentMode.CHARGING
        self.state.charging_steps += 1
        self.state.battery_level = min(
            self.battery_capacity,
            self.state.battery_level + self.recharge_rate,
        )

        if self.state.battery_level >= self.battery_capacity:
            self.state.current_goal = None
            self.state.current_path = []
            self.state.target_type = None
            self.state.mode = AgentMode.IDLE
        return True

    def perceive(self, env_map: GridMap) -> Observation:
        return self.sensor_model.sense(env_map, self.state.position)

    def update_belief(self, observation: Observation) -> None:
        for pos in observation.free_cells:
            self.belief_map.update_cell(pos, OccupancyState.FREE)
        for pos in observation.occupied_cells:
            self.belief_map.update_cell(pos, OccupancyState.OCCUPIED)
        for pos in observation.dynamic_cells:
            self.belief_map.update_cell(pos, OccupancyState.OCCUPIED)
        for pos in observation.charging_cells:
            self.state.discovered_charging_stations.add(pos)
            self.belief_map.update_cell(pos, OccupancyState.FREE)

    def plan_path(self, goal: Position) -> List[Position]:
        return self.planner.plan(self.state.position, goal, self.belief_map)

    def _has_enough_battery_for_plan(self, path_to_goal: List[Position], goal: Position) -> bool:
        if not self.enable_battery or self.state.battery_level is None:
            return True

        if not path_to_goal:
            return False

        # Important fix: use an actual known path from the candidate goal back to
        # a discovered charger. Do not use a Manhattan lower bound through
        # unknown space; that made robots over-confident and run out of battery.
        return_path_lengths: List[int] = []
        for charger in self._known_chargers():
            return_path = self._plan_between_known_cells(goal, charger)
            if return_path:
                return_path_lengths.append(self._path_move_count(return_path))

        if not return_path_lengths:
            return False

        outbound_moves = self._path_move_count(path_to_goal)
        return_moves = min(return_path_lengths)
        estimated_energy = (
            self._travel_energy_for_moves(outbound_moves, include_cleaning=True)
            + self._travel_energy_for_moves(return_moves, include_cleaning=False)
            + self.battery_safety_margin
        )
        return self.state.battery_level >= estimated_energy

    def _best_reachable_from_candidates(
        self,
        candidates: List[Position],
        reserved_goals: Optional[Set[Position]],
    ) -> tuple[Optional[Position], List[Position]]:
        reserved_goals = reserved_goals or set()
        candidates = sorted(
            [p for p in candidates if p not in reserved_goals or p == self.state.current_goal],
            key=lambda p: self._manhattan_distance(self.state.position, p),
        )
        for candidate in candidates[:80]:
            path = self.plan_path(candidate)
            if candidate == self.state.position:
                path = [candidate]
            if len(path) >= 1 and self._has_enough_battery_for_plan(path, candidate):
                return candidate, path
        return None, []

    def choose_reachable_goal(
        self,
        env_map: GridMap,
        reserved_goals: Optional[Set[Position]] = None,
    ) -> tuple[Optional[Position], List[Position], str, Optional[str]]:
        """
        Choose a reachable goal.

        Priority order:
        1. return to a discovered charger when battery is low or only enough to return;
        2. reachable frontier with enough battery to return afterwards;
        3. reachable known-free uncleaned cell with enough battery to return afterwards;
        4. return to charger when targets exist but are not battery-feasible.
        """
        if self.enable_battery and self._should_return_to_charge_now():
            charger, charger_path = self._nearest_reachable_charger_path(self.state.position)
            if charger is not None:
                if self._battery_is_low():
                    self.state.low_battery_returns += 1
                    reason = "low_battery_return"
                else:
                    self.state.battery_budget_returns += 1
                    reason = "battery_budget_return"
                self.state.assigned_charging_station = charger
                return charger, charger_path, reason, "charger"
            return None, [], "battery_low_but_no_discovered_reachable_charger", None

        frontiers = self.frontier_detector.detect(self.belief_map)
        goal, path = self._best_reachable_from_candidates(frontiers, reserved_goals)
        if goal is not None:
            return goal, path, "reachable_frontier_found", "frontier"

        dirty_known_cells = self.belief_map.known_free_uncleaned_cells()
        goal, path = self._best_reachable_from_candidates(dirty_known_cells, reserved_goals)
        if goal is not None:
            return goal, path, "reachable_cleaning_target_found", "known_dirty"

        if self.enable_battery:
            charger, charger_path = self._nearest_reachable_charger_path(self.state.position)
            if charger is not None and self.state.position != charger:
                self.state.battery_budget_returns += 1
                self.state.assigned_charging_station = charger
                return charger, charger_path, "battery_budget_return", "charger"

        if frontiers or dirty_known_cells:
            return None, [], "no_battery_feasible_or_reachable_target", None
        return None, [], "no_frontier_or_dirty_cell_available", None

    def mark_current_cell_cleaned(self, env_map: GridMap) -> None:
        pos = self.state.position
        newly_cleaned = env_map.mark_cleaned(pos, robot_id=self.state.robot_id)
        self.belief_map.mark_cleaned(pos)
        if newly_cleaned:
            self.state.cleaned_cells += 1
            self._consume_energy(self.cleaning_energy_cost)

    def _goal_needs_refresh(self, reserved_goals: Optional[Set[Position]]) -> bool:
        if self.state.current_goal is None:
            return True
        if self.state.position == self.state.current_goal:
            return True
        if not self.state.current_path:
            return True
        if reserved_goals and self.state.current_goal in reserved_goals and self.state.target_type != "charger":
            return True
        # If a non-charger goal becomes unsafe due to energy consumption, replan.
        if self.enable_battery and self.state.target_type != "charger" and self._should_return_to_charge_now():
            return True
        return False

    def _is_on_any_charger(self, env_map: GridMap) -> bool:
        return self.state.position in env_map.charging_stations

    def step(
        self,
        env_map: GridMap,
        reserved_goals: Optional[Set[Position]] = None,
        can_charge: bool = True,
    ) -> None:
        if self.state.mode == AgentMode.DONE:
            return

        self.state.steps_taken += 1

        # Home charger is always known. Public chargers are discovered by sensing.
        if self.state.charging_station is not None:
            self.state.discovered_charging_stations.add(self.state.charging_station)
            self.belief_map.update_cell(self.state.charging_station, OccupancyState.FREE)

        # If at a charging station and not full, spend the step charging or waiting.
        if self.enable_battery and self._is_on_any_charger(env_map):
            # Standing on a station means it is discovered, even if it was unknown earlier.
            self.state.discovered_charging_stations.add(self.state.position)
            self.belief_map.update_cell(self.state.position, OccupancyState.FREE)
            if self._recharge_if_possible(can_charge=can_charge):
                return

        observation = self.perceive(env_map)
        self._consume_energy(self.sensing_energy_cost)
        self.update_belief(observation)
        self.belief_map.update_cell(self.state.position, OccupancyState.FREE)

        self.mark_current_cell_cleaned(env_map)

        if self.enable_battery and self.state.battery_level is not None and self.state.battery_level <= 0:
            if self._is_on_any_charger(env_map):
                self._recharge_if_possible(can_charge=can_charge)
                return
            self.state.mode = AgentMode.DONE
            self.state.done_reason = "battery_depleted"
            self.state.battery_depletion_count += 1
            return

        if self._goal_needs_refresh(reserved_goals):
            goal, path, reason, target_type = self.choose_reachable_goal(env_map, reserved_goals)
            self.state.current_goal = goal
            self.state.current_path = path
            self.state.target_type = target_type

            if goal is None:
                self.state.mode = AgentMode.DONE
                self.state.done_reason = reason
                return

            if goal != self.state.position and len(path) < 2:
                self._debug_planning_failure(env_map, goal)

        if len(self.state.current_path) >= 2:
            self.state.mode = (
                AgentMode.RETURNING_TO_CHARGE
                if self.state.target_type == "charger"
                else AgentMode.MOVING
            )
            action = self.executor.next_action_from_path(
                self.state.current_path,
                self.state.position,
            )
            new_pos = self.executor.execute_move(self.state.position, action, env_map)

            if new_pos == self.state.position:
                self.state.mode = AgentMode.BLOCKED
                self.state.idle_steps += 1
                self.state.current_path = []
            else:
                self.state.total_path_length += 1.0
                self._consume_energy(self.move_energy_cost)
                self.state.position = new_pos
                self.state.trajectory.append(new_pos)
                self.state.current_path = self.state.current_path[1:]
        else:
            if self.state.target_type == "charger" and self.enable_battery:
                if self._is_on_any_charger(env_map):
                    self._recharge_if_possible(can_charge=can_charge)
                else:
                    self.state.mode = AgentMode.RETURNING_TO_CHARGE
            else:
                self.state.mode = AgentMode.IDLE
                self.state.idle_steps += 1
            self.state.current_path = []

        self.mark_current_cell_cleaned(env_map)

        if self.enable_battery and self.state.battery_level is not None and self.state.battery_level <= 0:
            if not self._is_on_any_charger(env_map):
                self.state.mode = AgentMode.DONE
                self.state.done_reason = "battery_depleted"
                self.state.battery_depletion_count += 1
