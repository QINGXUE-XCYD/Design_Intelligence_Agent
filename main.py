from config.schema import SimulationConfig
from environment.map_generator import MapGenerator
from mapping.occupancy_grid import OccupancyGrid
from sensing.sensor_model import SensorModel
from exploration.frontier_detector import FrontierDetector
from planning.astar_planner import AStarPlanner
from control.executor import ActionExecutor
from agents.robot_agent import RobotAgent
from simulation.engine import SimulationEngine
from metrics.collector import MetricsCollector


def build_single_robot_demo() -> SimulationEngine:
    """
    构建单机器人演示系统 / Build a single-robot demo system
    """
    config = SimulationConfig()

    map_generator = MapGenerator(config.map_config)
    env_map = map_generator.generate()

    starts = map_generator.sample_robot_starts(
        env_map,
        config.robot_config.num_agents
    )

    agents = []
    for i, start_pos in enumerate(starts):
        belief_map = OccupancyGrid(
            width=config.map_config.width,
            height=config.map_config.height,
        )

        agent = RobotAgent(
            robot_id=i,
            start_pos=start_pos,
            belief_map=belief_map,
            sensor_model=SensorModel(config.robot_config.sensor_range),
            frontier_detector=FrontierDetector(),
            planner=AStarPlanner(),
            executor=ActionExecutor(),
        )
        agents.append(agent)

    metrics = MetricsCollector()

    engine = SimulationEngine(
        env_map=env_map,
        agents=agents,
        metrics_collector=metrics,
        max_steps=config.robot_config.max_steps,
    )
    return engine


if __name__ == "__main__":
    engine = build_single_robot_demo()
    results = engine.run()
    print("Simulation finished.")
    print(results)