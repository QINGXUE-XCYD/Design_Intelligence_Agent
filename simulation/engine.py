from typing import List

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