from dataclasses import asdict
from datetime import datetime
from collections import Counter
from copy import deepcopy
from statistics import mean, stdev

from agents.agent_state import AgentRole
from agents.robot_agent import RobotAgent
from config.schema import SimulationConfig
from control.executor import ActionExecutor
from coordination.reservation_table import GoalReservationTable
from environment.map_generator import MapGenerator
from exploration.frontier_detector import FrontierDetector
from mapping.occupancy_grid import OccupancyGrid
from metrics.collector import MetricsCollector
from planning.astar_planner import AStarPlanner
from sensing.sensor_model import SensorModel
from simulation.engine import SimulationEngine
from utils.showcase_builder import RunShowcaseBuilder
from utils.run_saver import RunSaver
from visualisation.renderer import MapRenderer


def build_engine_from_config(config: SimulationConfig) -> SimulationEngine:
    """
    根据给定配置构建仿真引擎
    Build simulation engine from a given config
    """
    if config.robot_config.num_agents != 3:
        raise ValueError("The fixed-role baseline currently requires exactly 3 robots.")

    if not config.coordination_config.release_cleaners_after_scout:
        raise ValueError("The current baseline requires cleaners to start after scouting.")

    scout_id = config.coordination_config.scout_robot_id
    cleaner_ids = config.coordination_config.cleaner_robot_ids
    expected_cleaner_ids = [robot_id for robot_id in range(config.robot_config.num_agents) if robot_id != scout_id]
    if sorted(cleaner_ids) != expected_cleaner_ids:
        raise ValueError("Cleaner robot ids must cover every non-scout robot in the fixed-role baseline.")

    map_generator = MapGenerator(config.map_config)
    env_map = map_generator.generate()

    starts = map_generator.sample_robot_starts(
        env_map,
        config.robot_config.num_agents
    )

    shared_belief_map = OccupancyGrid(
        width=config.map_config.width,
        height=config.map_config.height,
    )
    reservation_table = (
        GoalReservationTable()
        if config.coordination_config.enable_goal_reservation
        else None
    )

    agents = []
    for i, start_pos in enumerate(starts):
        role = AgentRole.SCOUT if i == scout_id else AgentRole.CLEANER
        is_active = role == AgentRole.SCOUT

        agent = RobotAgent(
            robot_id=i,
            start_pos=start_pos,
            belief_map=shared_belief_map,
            sensor_model=SensorModel(config.robot_config.sensor_range),
            frontier_detector=FrontierDetector(),
            planner=AStarPlanner(),
            executor=ActionExecutor(),
            role=role,
            active=is_active,
            reservation_table=reservation_table,
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


def build_single_robot_demo() -> tuple[SimulationEngine, SimulationConfig]:
    """
    保留原有辅助入口名，但当前返回固定三机器人协同系统
    Preserve the helper name while returning the fixed-role three-robot system
    """
    config = SimulationConfig()
    engine = build_engine_from_config(config)
    return engine, config


def build_run_metadata(config: SimulationConfig, engine: SimulationEngine, run_dir_name: str) -> dict:
    """
    构建单次运行元数据 / Build metadata for a single run
    """
    per_agent_summary = engine.export_per_agent_summary()

    return {
        "run_id": run_dir_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": config.map_config.seed,
        "num_agents": config.robot_config.num_agents,
        "coordination_mode": "fixed_role_three_robot",
        "cleaning_phase_start_step": engine.cleaning_phase_start_step,
        "config": asdict(config),
        "agent_start_positions": {
            item["robot_id"]: item["start_pos"]
            for item in per_agent_summary
        },
        "agent_final_positions": {
            item["robot_id"]: item["final_pos"]
            for item in per_agent_summary
        },
    }


def build_seed_result(seed: int, results: dict, engine: SimulationEngine) -> dict:
    """
    构建单个 seed 的汇总结果
    Build one-row summary result for a single seed
    """
    success = (
            results.get("termination_reason") != "max_steps_reached"
            and results.get("coverage_rate", 0.0) >= 0.99
    )

    return {
        "seed": seed,
        "num_agents": len(engine.agents),
        "total_steps": results.get("total_steps"),
        "coverage_rate": results.get("coverage_rate"),
        "total_path_length": results.get("total_path_length"),
        "total_cleaned_cells": results.get("total_cleaned_cells"),
        "idle_steps": results.get("idle_steps"),
        "termination_reason": results.get("termination_reason"),
        "success": success,
    }


def aggregate_batch_results(seed_results: list[dict]) -> dict:
    """
    聚合多 seed 运行结果
    Aggregate results across multiple seeds
    """

    def safe_mean(values: list[float]) -> float:
        return mean(values) if values else 0.0

    def safe_std(values: list[float]) -> float:
        return stdev(values) if len(values) > 1 else 0.0

    steps = [row["total_steps"] for row in seed_results]
    coverage = [row["coverage_rate"] for row in seed_results]
    path_lengths = [row["total_path_length"] for row in seed_results]
    idle_steps = [row["idle_steps"] for row in seed_results]
    successes = [row["success"] for row in seed_results]
    termination_reasons = [row["termination_reason"] for row in seed_results]

    return {
        "num_runs": len(seed_results),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,

        "mean_steps": safe_mean(steps),
        "std_steps": safe_std(steps),
        "min_steps": min(steps) if steps else 0,
        "max_steps": max(steps) if steps else 0,

        "mean_coverage": safe_mean(coverage),
        "std_coverage": safe_std(coverage),
        "min_coverage": min(coverage) if coverage else 0.0,
        "max_coverage": max(coverage) if coverage else 0.0,

        "mean_path_length": safe_mean(path_lengths),
        "std_path_length": safe_std(path_lengths),
        "min_path_length": min(path_lengths) if path_lengths else 0.0,
        "max_path_length": max(path_lengths) if path_lengths else 0.0,

        "mean_idle_steps": safe_mean(idle_steps),
        "std_idle_steps": safe_std(idle_steps),

        "termination_reason_distribution": dict(Counter(termination_reasons)),
    }


def run_batch_experiment(config: SimulationConfig) -> None:
    """
    运行多 seed 批量实验
    Run a multi-seed batch experiment
    """
    exp_dir = RunSaver.create_experiment_dir(
        base_dir=config.batch_config.base_output_dir,
        experiment_name=config.batch_config.experiment_name,
    )

    seed_results: list[dict] = []

    for seed in config.batch_config.seeds:
        run_config = deepcopy(config)
        run_config.map_config.seed = seed

        engine = build_engine_from_config(run_config)
        results = engine.run()

        seed_result = build_seed_result(seed, results, engine)
        seed_results.append(seed_result)

        print(f"[BATCH] seed={seed} finished -> {seed_result}")

    aggregate_summary = aggregate_batch_results(seed_results)

    RunSaver.save_json(asdict(config), exp_dir / "experiment_config.json")
    RunSaver.save_step_records(seed_results, exp_dir / "seed_results.csv")
    RunSaver.save_json(aggregate_summary, exp_dir / "aggregate_summary.json")

    print(f"[BATCH] experiment finished. Results saved to: {exp_dir}")


if __name__ == "__main__":
    config = SimulationConfig()

    if config.batch_config.enabled:
        run_batch_experiment(config)
    else:
        engine = build_engine_from_config(config)

        run_dir = RunSaver.create_run_dir()
        renderer = MapRenderer()

        results = engine.run()

        run_metadata = build_run_metadata(config, engine, run_dir.name)
        per_agent_summary = engine.export_per_agent_summary()
        agent_trajectories = engine.export_agent_trajectories()
        global_step_records = engine.metrics_collector.export_step_records()
        agent_step_records = engine.export_agent_step_records()

        RunSaver.save_json(run_metadata, run_dir / "run_metadata.json")
        RunSaver.save_metrics(results, run_dir / "metrics.json")
        RunSaver.save_json(per_agent_summary, run_dir / "per_agent_summary.json")
        RunSaver.save_json(agent_trajectories, run_dir / "agent_trajectories.json")
        RunSaver.save_step_records(global_step_records, run_dir / "step_records.csv")
        RunSaver.save_step_records(agent_step_records, run_dir / "agent_step_log.csv")

        renderer.render_run_overview(
            env_map=engine.env_map,
            agents=engine.agents,
            save_path=run_dir / "run_overview.png",
        )

        renderer.plot_coverage_curve(
            global_step_records,
            run_dir / "coverage_curve.png",
        )

        showcase_path = RunShowcaseBuilder.build_run_showcase(run_dir)

        print("Simulation finished.")
        print(results)
        print(f"Artifacts saved to: {run_dir}")
        print(f"Showcase saved to: {showcase_path}")
