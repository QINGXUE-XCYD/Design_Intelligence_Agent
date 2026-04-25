from __future__ import annotations
from typing import Dict, List, Optional

from agents.agent_state import AgentMode
from agents.robot_agent import RobotAgent
from environment.grid_map import GridMap, Position


class MetricsCollector:
    """
    指标收集器 / Metrics collector.
    """

    def __init__(self) -> None:
        self.step_records: List[Dict] = []
        self.agent_step_records: List[Dict] = []

    def record_step(self, env_map: GridMap, agents: List[RobotAgent], step_idx: int) -> None:
        coverage_rate = self.compute_coverage_rate(env_map)
        self.step_records.append({
            "step": step_idx,
            "coverage_rate": coverage_rate,
            "total_path_length": self.compute_total_path_length(agents),
            "total_cleaned_cells": len(env_map.cleaned_cells),
            "duplicate_visit_count": self.compute_duplicate_visit_count(agents),
            "inter_agent_overlap_cells": self.compute_inter_agent_overlap_cells(agents),
            "active_agents": sum(a.state.mode != AgentMode.DONE for a in agents),
            "mean_battery_level": self.compute_mean_battery_level(agents),
            "min_battery_level": self.compute_min_battery_level(agents),
            "agents_charging": sum(a.state.mode == AgentMode.CHARGING for a in agents),
            "agents_waiting_for_charge": sum(a.state.mode == AgentMode.WAITING_FOR_CHARGE for a in agents),
            "total_charging_steps_so_far": sum(a.state.charging_steps for a in agents),
            "total_charge_wait_steps_so_far": sum(a.state.charge_wait_steps for a in agents),
            "total_energy_used_so_far": sum(a.state.energy_used for a in agents),
        })

        for agent in agents:
            goal = agent.state.current_goal
            self.agent_step_records.append({
                "step": step_idx,
                "robot_id": agent.state.robot_id,
                "x": agent.state.position[0],
                "y": agent.state.position[1],
                "mode": agent.state.mode.value,
                "goal_x": goal[0] if goal is not None else "",
                "goal_y": goal[1] if goal is not None else "",
                "target_type": agent.state.target_type or "",
                "battery_level": agent.state.battery_level if agent.state.battery_level is not None else "",
                "battery_percent": (
                    agent.state.battery_level / agent.battery_capacity
                    if agent.state.battery_level is not None and agent.battery_capacity > 0
                    else ""
                ),
                "cleaned_cells": agent.state.cleaned_cells,
                "path_length": agent.state.total_path_length,
                "idle_steps": agent.state.idle_steps,
                "charging_steps": agent.state.charging_steps,
                "charging_events": agent.state.charging_events,
                "charge_wait_steps": agent.state.charge_wait_steps,
                "low_battery_returns": agent.state.low_battery_returns,
                "battery_budget_returns": agent.state.battery_budget_returns,
                "battery_depletion_count": agent.state.battery_depletion_count,
                "energy_used": agent.state.energy_used,
                "discovered_charger_count": len(agent.state.discovered_charging_stations),
                "done_reason": agent.state.done_reason or "",
            })

    def export_step_records(self) -> List[Dict]:
        return list(self.step_records)

    def export_agent_step_records(self) -> List[Dict]:
        return list(self.agent_step_records)

    def finalize_episode(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        total_steps: int,
        target_coverage: float = 0.95,
    ) -> Dict:
        final_coverage = self.compute_coverage_rate(env_map)
        return {
            "total_steps": total_steps,
            "coverage_rate": final_coverage,
            "target_coverage": target_coverage,
            "success": final_coverage >= target_coverage,
            "steps_to_80_coverage": self.steps_to_coverage(0.80),
            "steps_to_90_coverage": self.steps_to_coverage(0.90),
            "steps_to_95_coverage": self.steps_to_coverage(0.95),
            "total_path_length": self.compute_total_path_length(agents),
            "total_cleaned_cells": len(env_map.cleaned_cells),
            "total_cleanable_cells": env_map.count_cleanable_cells(),
            "idle_steps": sum(a.state.idle_steps for a in agents),
            "duplicate_visit_count": self.compute_duplicate_visit_count(agents),
            "inter_agent_overlap_cells": self.compute_inter_agent_overlap_cells(agents),
            "final_mean_battery_level": self.compute_mean_battery_level(agents),
            "final_min_battery_level": self.compute_min_battery_level(agents),
            "total_energy_used": sum(a.state.energy_used for a in agents),
            "total_charging_steps": sum(a.state.charging_steps for a in agents),
            "total_charging_events": sum(a.state.charging_events for a in agents),
            "total_charge_wait_steps": sum(a.state.charge_wait_steps for a in agents),
            "total_low_battery_returns": sum(a.state.low_battery_returns for a in agents),
            "total_battery_budget_returns": sum(a.state.battery_budget_returns for a in agents),
            "total_battery_depletion_count": sum(a.state.battery_depletion_count for a in agents),
            "per_agent_path_length": {
                str(a.state.robot_id): a.state.total_path_length for a in agents
            },
            "per_agent_cleaned_cells": {
                str(a.state.robot_id): a.state.cleaned_cells for a in agents
            },
            "per_agent_final_battery": {
                str(a.state.robot_id): a.state.battery_level for a in agents
            },
            "per_agent_charging_steps": {
                str(a.state.robot_id): a.state.charging_steps for a in agents
            },
            "per_agent_charge_wait_steps": {
                str(a.state.robot_id): a.state.charge_wait_steps for a in agents
            },
        }

    def per_agent_summary(self, agents: List[RobotAgent]) -> List[Dict]:
        summaries: List[Dict] = []
        for agent in agents:
            summaries.append({
                "robot_id": agent.state.robot_id,
                "start_position": list(agent.state.trajectory[0]) if agent.state.trajectory else None,
                "final_position": list(agent.state.position),
                "mode": agent.state.mode.value,
                "done_reason": agent.state.done_reason,
                "steps_taken": agent.state.steps_taken,
                "cleaned_cells": agent.state.cleaned_cells,
                "idle_steps": agent.state.idle_steps,
                "total_path_length": agent.state.total_path_length,
                "trajectory_length": len(agent.state.trajectory),
                "battery_level": agent.state.battery_level,
                "energy_used": agent.state.energy_used,
                "home_charging_station": list(agent.state.charging_station) if agent.state.charging_station else None,
                "assigned_charging_station": (
                    list(agent.state.assigned_charging_station)
                    if agent.state.assigned_charging_station else None
                ),
                "discovered_charging_stations": [
                    list(pos) for pos in sorted(agent.state.discovered_charging_stations)
                ],
                "charging_steps": agent.state.charging_steps,
                "charging_events": agent.state.charging_events,
                "charge_wait_steps": agent.state.charge_wait_steps,
                "low_battery_returns": agent.state.low_battery_returns,
                "battery_budget_returns": agent.state.battery_budget_returns,
                "battery_depletion_count": agent.state.battery_depletion_count,
            })
        return summaries

    def agent_trajectories(self, agents: List[RobotAgent]) -> Dict[str, List[List[int]]]:
        return {
            str(agent.state.robot_id): [list(p) for p in agent.state.trajectory]
            for agent in agents
        }

    def compute_coverage_rate(self, env_map: GridMap) -> float:
        total_cleanable = env_map.count_cleanable_cells()
        if total_cleanable == 0:
            return 0.0
        return len(env_map.cleaned_cells) / total_cleanable

    def compute_total_path_length(self, agents: List[RobotAgent]) -> float:
        return sum(agent.state.total_path_length for agent in agents)

    def compute_duplicate_visit_count(self, agents: List[RobotAgent]) -> int:
        all_visits: List[Position] = []
        for agent in agents:
            all_visits.extend(agent.state.trajectory)
        return len(all_visits) - len(set(all_visits))

    def compute_inter_agent_overlap_cells(self, agents: List[RobotAgent]) -> int:
        visit_sets = [set(agent.state.trajectory) for agent in agents]
        if len(visit_sets) < 2:
            return 0
        cell_counts: Dict[Position, int] = {}
        for visit_set in visit_sets:
            for cell in visit_set:
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
        return sum(1 for count in cell_counts.values() if count > 1)

    def compute_mean_battery_level(self, agents: List[RobotAgent]) -> Optional[float]:
        levels = [a.state.battery_level for a in agents if a.state.battery_level is not None]
        if not levels:
            return None
        return sum(levels) / len(levels)

    def compute_min_battery_level(self, agents: List[RobotAgent]) -> Optional[float]:
        levels = [a.state.battery_level for a in agents if a.state.battery_level is not None]
        if not levels:
            return None
        return min(levels)

    def steps_to_coverage(self, threshold: float) -> Optional[int]:
        for record in self.step_records:
            if record["coverage_rate"] >= threshold:
                return int(record["step"])
        return None
