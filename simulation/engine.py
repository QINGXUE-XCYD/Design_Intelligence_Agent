from typing import Dict, List

from agents.agent_state import AgentMode
from agents.robot_agent import RobotAgent
from environment.grid_map import GridMap
from metrics.collector import MetricsCollector


class SimulationEngine:
    """
    仿真主引擎 / Main simulation engine

    负责控制每个 step 的执行顺序。
    Responsible for controlling the execution order of each simulation step.
    """

    def __init__(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        metrics_collector: MetricsCollector,
        max_steps: int,
    ) -> None:
        self.env_map = env_map
        self.agents = agents
        self.metrics_collector = metrics_collector
        self.max_steps = max_steps
        self.current_step = 0
        self.termination_reason = None

        # 逐 robot 的 step 级日志 / Per-robot step-level logs
        self.agent_step_records: List[Dict] = []

        # 记录初始状态 / Record initial state snapshot
        self._record_agent_step_snapshot(step_idx=-1, record_type="initial")

    def _record_agent_step_snapshot(self, step_idx: int, record_type: str = "step") -> None:
        """
        记录每个 step、每个机器人的详细状态
        Record detailed per-step state for every robot
        """
        for agent in self.agents:
            goal = agent.state.current_goal
            pos = agent.state.position

            self.agent_step_records.append({
                "record_type": record_type,
                "step": step_idx,
                "robot_id": agent.state.robot_id,
                "x": pos[0],
                "y": pos[1],
                "mode": agent.state.mode.value if hasattr(agent.state.mode, "value") else str(agent.state.mode),
                "goal_x": goal[0] if goal is not None else None,
                "goal_y": goal[1] if goal is not None else None,
                "path_remaining": len(agent.state.current_path),
                "trajectory_length": len(agent.state.trajectory),
                "cleaned_cells": agent.state.cleaned_cells,
                "idle_steps": agent.state.idle_steps,
                "in_cleanup_phase": getattr(agent, "in_cleanup_phase", False),
                "done_reason": agent.state.done_reason,
            })

    def export_agent_step_records(self) -> List[Dict]:
        """
        导出逐 robot 的 step 日志 / Export per-robot step logs
        """
        return self.agent_step_records

    def export_per_agent_summary(self) -> List[Dict]:
        """
        导出每个机器人的汇总信息 / Export per-agent summary
        """
        summaries: List[Dict] = []

        for agent in self.agents:
            traj = agent.state.trajectory
            start_pos = traj[0] if traj else agent.state.position
            final_pos = traj[-1] if traj else agent.state.position
            cleanup_cells = getattr(agent, "cleanup_cleaned_cells", set())

            summaries.append({
                "robot_id": agent.state.robot_id,
                "start_pos": list(start_pos),
                "final_pos": list(final_pos),
                "trajectory_length": len(traj),
                "path_length": max(0, len(traj) - 1),
                "cleaned_cells": agent.state.cleaned_cells,
                "idle_steps": agent.state.idle_steps,
                "mode": agent.state.mode.value if hasattr(agent.state.mode, "value") else str(agent.state.mode),
                "done_reason": agent.state.done_reason,
                "cleanup_started": getattr(agent, "in_cleanup_phase", False),
                "cleanup_start_traj_index": getattr(agent, "cleanup_start_traj_index", None),
                "cleanup_cleaned_cells_count": len(cleanup_cells),
            })

        return summaries

    def export_agent_trajectories(self) -> List[Dict]:
        """
        导出每个机器人的完整轨迹和阶段信息
        Export full per-agent trajectories and phase information
        """
        data: List[Dict] = []

        for agent in self.agents:
            cleanup_cells = sorted(list(getattr(agent, "cleanup_cleaned_cells", set())))
            data.append({
                "robot_id": agent.state.robot_id,
                "trajectory": [list(pos) for pos in agent.state.trajectory],
                "cleanup_start_traj_index": getattr(agent, "cleanup_start_traj_index", None),
                "cleanup_cleaned_cells": [list(pos) for pos in cleanup_cells],
            })

        return data

    def step(self) -> None:
        """
        执行一个仿真 step / Execute one simulation step
        """
        for agent in self.agents:
            agent.step(self.env_map)

        self.metrics_collector.record_step(
            env_map=self.env_map,
            agents=self.agents,
            step_idx=self.current_step,
        )
        self._record_agent_step_snapshot(step_idx=self.current_step, record_type="step")
        self.current_step += 1

    def is_done(self) -> bool:
        """
        判断仿真是否结束 / Check whether simulation is finished
        """
        if self.current_step >= self.max_steps:
            self.termination_reason = "max_steps_reached"
            return True

        if all(agent.state.mode == AgentMode.DONE for agent in self.agents):
            self.termination_reason = "all_agents_done"
            return True

        return False

    def run(self) -> dict:
        """
        运行仿真直到结束 / Run simulation until termination
        """
        while not self.is_done():
            self.step()

        results = self.metrics_collector.finalize_episode(
            env_map=self.env_map,
            agents=self.agents,
            total_steps=self.current_step,
        )

        results["termination_reason"] = self.termination_reason
        results["per_agent_done_reason"] = {
            agent.state.robot_id: agent.state.done_reason
            for agent in self.agents
        }

        return results