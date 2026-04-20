from typing import Dict, List

from environment.grid_map import GridMap
from agents.robot_agent import RobotAgent


class MetricsCollector:
    """
    指标收集器 / Metrics collector

    用于记录每一步和整局仿真的性能指标。
    Used to record per-step and per-episode metrics.
    """

    def __init__(self) -> None:
        self.step_records: List[Dict] = []

    def record_step(self, env_map: GridMap, agents: List[RobotAgent], step_idx: int) -> None:
        """
        记录单步指标 / Record per-step metrics
        """
        self.step_records.append({
            "step": step_idx,
            "coverage_rate": self.compute_coverage_rate(env_map),
            "total_path_length": self.compute_total_path_length(agents),
        })

    def finalize_episode(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        total_steps: int,
    ) -> Dict:
        """
        汇总整局仿真结果 / Summarize final episode results
        """
        return {
            "total_steps": total_steps,
            "coverage_rate": self.compute_coverage_rate(env_map),
            "total_path_length": self.compute_total_path_length(agents),
            "total_cleaned_cells": sum(a.state.cleaned_cells for a in agents),
            "idle_steps": sum(a.state.idle_steps for a in agents),
        }

    def compute_coverage_rate(self, env_map: GridMap) -> float:
        """
        计算覆盖率 / Compute coverage rate
        """
        total_cleanable = env_map.count_cleanable_cells()
        if total_cleanable == 0:
            return 0.0
        return len(env_map.cleaned_cells) / total_cleanable

    def compute_total_path_length(self, agents: List[RobotAgent]) -> float:
        """
        计算总路径长度 / Compute total path length
        """
        return sum(agent.state.total_path_length for agent in agents)