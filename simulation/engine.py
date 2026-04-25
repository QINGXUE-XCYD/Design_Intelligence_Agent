from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from agents.agent_state import AgentMode
from agents.robot_agent import RobotAgent
from environment.dynamic_obstacles import RandomWalkDynamicObstaclePolicy
from environment.grid_map import GridMap, Position
from mapping.occupancy_grid import OccupancyGrid
from metrics.collector import MetricsCollector


class SimulationEngine:
    """
    仿真主引擎 / Main simulation engine.
    """

    def __init__(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        metrics_collector: MetricsCollector,
        max_steps: int,
        target_coverage: float = 0.95,
        coordination_strategy: str = "independent",
        communication_interval: int = 1,
        communication_loss_rate: float = 0.0,
        charging_station_capacity: int = 1,
        dynamic_obstacle_policy: Optional[RandomWalkDynamicObstaclePolicy] = None,
        random_seed: int = 0,
    ) -> None:
        self.env_map = env_map
        self.agents = agents
        self.metrics_collector = metrics_collector
        self.max_steps = max_steps
        self.target_coverage = target_coverage
        self.coordination_strategy = coordination_strategy
        self.communication_interval = max(1, communication_interval)
        self.communication_loss_rate = max(0.0, min(1.0, communication_loss_rate))
        self.charging_station_capacity = max(1, charging_station_capacity)
        self.dynamic_obstacle_policy = dynamic_obstacle_policy
        self.rng = random.Random(random_seed)

        self.current_step = 0
        self.termination_reason: Optional[str] = None
        self.shared_belief_map: Optional[OccupancyGrid] = None
        self.shared_discovered_chargers: Set[Position] = set()
        if "shared_map" in coordination_strategy:
            self.shared_belief_map = OccupancyGrid(env_map.width, env_map.height)

    def step(self) -> None:
        """
        执行一个仿真 step / Execute one simulation step.
        """
        robot_positions = {agent.state.position for agent in self.agents if agent.state.mode != AgentMode.DONE}
        if self.dynamic_obstacle_policy is not None:
            self.dynamic_obstacle_policy.step(self.env_map, forbidden=robot_positions)

        self._communicate_maps()

        selected_goals: Set[Position] = set()
        charge_permissions = self._charging_permissions()

        for agent in self.agents:
            if agent.state.mode == AgentMode.DONE:
                continue

            reserved_goals = None
            if self._use_goal_reservation():
                reserved_goals = self._reserved_goals_for(agent, selected_goals)

            agent.step(
                self.env_map,
                reserved_goals=reserved_goals,
                can_charge=charge_permissions.get(agent.state.robot_id, True),
            )

            if agent.state.current_goal is not None:
                selected_goals.add(agent.state.current_goal)

        self._communicate_maps()

        self.metrics_collector.record_step(
            env_map=self.env_map,
            agents=self.agents,
            step_idx=self.current_step,
        )
        self.current_step += 1

    def _use_goal_reservation(self) -> bool:
        return "reservation" in self.coordination_strategy or self.coordination_strategy == "goal_reservation"

    def _reserved_goals_for(self, current_agent: RobotAgent, selected_goals: Set[Position]) -> Set[Position]:
        goals = set(selected_goals)
        for agent in self.agents:
            if agent is current_agent or agent.state.mode == AgentMode.DONE:
                continue
            if agent.state.current_goal is not None:
                goals.add(agent.state.current_goal)
        return goals

    def _charging_permissions(self) -> Dict[int, bool]:
        """
        Enforce charging station capacity.

        Robots physically at the same charging station compete for limited
        charging slots. Lowest battery is prioritised.
        """
        permissions: Dict[int, bool] = {
            agent.state.robot_id: True for agent in self.agents
        }

        station_to_agents: Dict[Position, List[RobotAgent]] = defaultdict(list)
        for agent in self.agents:
            if (
                agent.enable_battery
                and agent.state.battery_level is not None
                and agent.state.position in self.env_map.charging_stations
                and agent.state.battery_level < agent.battery_capacity
                and agent.state.mode != AgentMode.DONE
            ):
                station_to_agents[agent.state.position].append(agent)

        for station, waiting_agents in station_to_agents.items():
            waiting_agents.sort(key=lambda a: (a.state.battery_level or 0.0, a.state.robot_id))
            allowed = {a.state.robot_id for a in waiting_agents[: self.charging_station_capacity]}
            for agent in waiting_agents:
                permissions[agent.state.robot_id] = agent.state.robot_id in allowed

        return permissions

    def _communicate_maps(self) -> None:
        if self.shared_belief_map is None:
            return
        if self.current_step % self.communication_interval != 0:
            return

        # Upload local maps and discovered chargers to global shared state.
        for agent in self.agents:
            if self.rng.random() < self.communication_loss_rate:
                continue
            self.shared_belief_map.merge(agent.belief_map)
            self.shared_discovered_chargers.update(agent.state.discovered_charging_stations)
            agent.consume_communication_energy()

        # Download global map and shared charger discoveries to each robot.
        for agent in self.agents:
            if self.rng.random() < self.communication_loss_rate:
                continue
            agent.belief_map.merge(self.shared_belief_map)
            agent.merge_discovered_chargers(self.shared_discovered_chargers)
            agent.consume_communication_energy()

    def is_done(self) -> bool:
        if self.current_step >= self.max_steps:
            self.termination_reason = "max_steps_reached"
            return True

        if self.metrics_collector.compute_coverage_rate(self.env_map) >= self.target_coverage:
            self.termination_reason = "target_coverage_reached"
            return True

        if all(agent.state.mode == AgentMode.DONE for agent in self.agents):
            self.termination_reason = "all_agents_done"
            return True

        return False

    def run(self) -> dict:
        while not self.is_done():
            self.step()

        results = self.metrics_collector.finalize_episode(
            env_map=self.env_map,
            agents=self.agents,
            total_steps=self.current_step,
            target_coverage=self.target_coverage,
        )
        results["termination_reason"] = self.termination_reason
        results["coordination_strategy"] = self.coordination_strategy
        results["charging_station_capacity"] = self.charging_station_capacity
        results["charging_station_count"] = len(self.env_map.charging_stations)
        results["per_agent_done_reason"] = {
            str(agent.state.robot_id): agent.state.done_reason
            for agent in self.agents
        }
        return results
