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
from utils.run_saver import RunSaver
from visualisation.renderer import MapRenderer


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

    run_dir = RunSaver.create_run_dir()
    renderer = MapRenderer()

    # 运行仿真 / Run simulation
    results = engine.run()

    # 保存结果 / Save artifacts
    RunSaver.save_metrics(results, run_dir / "metrics.json")
    RunSaver.save_step_records(
        engine.metrics_collector.export_step_records(),
        run_dir / "step_records.csv",
    )

    renderer.render_run_overview(
        env_map=engine.env_map,
        agents=engine.agents,
        save_path=run_dir / "run_overview.png",
    )
    
    renderer.plot_coverage_curve(
        engine.metrics_collector.export_step_records(),
        run_dir / "coverage_curve.png",
    )

    print("Simulation finished.")
    print(results)
    print(f"Artifacts saved to: {run_dir}")