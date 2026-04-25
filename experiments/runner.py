from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

from config.schema import SimulationConfig
from simulation.factory import build_simulation
from utils.run_saver import RunSaver
from visualisation.renderer import MapRenderer


SUMMARY_METRICS = [
    "coverage_rate",
    "success",
    "total_steps",
    "total_path_length",
    "total_cleaned_cells",
    "idle_steps",
    "duplicate_visit_count",
    "inter_agent_overlap_cells",
    "steps_to_80_coverage",
    "steps_to_90_coverage",
    "steps_to_95_coverage",
    "final_mean_battery_level",
    "final_min_battery_level",
    "total_energy_used",
    "total_charging_steps",
    "total_charging_events",
    "total_charge_wait_steps",
    "total_low_battery_returns",
    "total_battery_budget_returns",
    "total_battery_depletion_count",
]


def run_single_simulation(
    config: SimulationConfig,
    run_dir: str | Path | None = None,
    save_artifacts: bool = True,
) -> Tuple[Dict, Path | None]:
    """
    Run one simulation and optionally save all artifacts described in README.
    """
    engine = build_simulation(config)
    results = engine.run()

    if not save_artifacts:
        return results, None

    if run_dir is None:
        run_dir = RunSaver.create_run_dir(config.output_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    renderer = MapRenderer()

    RunSaver.save_metrics(results, run_dir / "metrics.json")
    RunSaver.save_json(
        {
            "config": asdict(config),
            "start_positions": [list(agent.state.trajectory[0]) for agent in engine.agents],
            "final_positions": [list(agent.state.position) for agent in engine.agents],
            "total_cleanable_cells": engine.env_map.count_cleanable_cells(),
            "static_obstacle_count": len(engine.env_map.static_obstacles),
            "dynamic_obstacle_count": len(engine.env_map.dynamic_obstacles),
            "home_charging_stations": [list(pos) for pos in sorted(engine.env_map.home_charging_stations)],
            "public_charging_stations": [list(pos) for pos in sorted(engine.env_map.public_charging_stations)],
            "all_charging_stations": [list(pos) for pos in sorted(engine.env_map.charging_stations)],
        },
        run_dir / "run_metadata.json",
    )
    RunSaver.save_json(
        engine.metrics_collector.per_agent_summary(engine.agents),
        run_dir / "per_agent_summary.json",
    )
    RunSaver.save_json(
        engine.metrics_collector.agent_trajectories(engine.agents),
        run_dir / "agent_trajectories.json",
    )
    RunSaver.save_step_records(
        engine.metrics_collector.export_step_records(),
        run_dir / "step_records.csv",
    )
    RunSaver.save_agent_step_records(
        engine.metrics_collector.export_agent_step_records(),
        run_dir / "agent_step_log.csv",
    )

    renderer.render_run_overview(
        env_map=engine.env_map,
        agents=engine.agents,
        save_path=run_dir / "run_overview.png",
    )
    step_records = engine.metrics_collector.export_step_records()
    agent_step_records = engine.metrics_collector.export_agent_step_records()

    renderer.plot_coverage_curve(
        step_records,
        run_dir / "coverage_curve.png",
    )
    renderer.plot_battery_curve(
        step_records,
        run_dir / "battery_curve.png",
    )
    renderer.render_cleaning_animation(
        env_map=engine.env_map,
        agents=engine.agents,
        step_records=step_records,
        agent_step_records=agent_step_records,
        save_path=run_dir / "cleaning_animation.gif",
        fps=4,
        max_frames=250,
    )
    return results, run_dir


class ExperimentRunner:
    """
    Batch experiment runner.

    It repeats runs over seeds and condition combinations, saves one row per
    seed, and writes aggregated descriptive statistics for each condition.
    """

    def __init__(self, base_config: SimulationConfig) -> None:
        self.base_config = base_config

    def iter_condition_configs(self) -> Iterable[SimulationConfig]:
        b = self.base_config.batch_config
        for num_agents, strategy, sensor_range, obstacle_density, seed in product(
            b.num_agents_values,
            b.strategies,
            b.sensor_ranges,
            b.obstacle_densities,
            b.seeds,
        ):
            config = deepcopy(self.base_config)
            config.batch_config.enabled = False
            config.robot_config.num_agents = num_agents
            config.robot_config.sensor_range = sensor_range
            config.map_config.obstacle_density = obstacle_density
            config.map_config.seed = seed
            config.coordination_config.strategy = strategy
            yield config

    def run(self, experiment_dir: str | Path | None = None) -> Path:
        if experiment_dir is None:
            experiment_dir = RunSaver.create_experiment_dir(
                base_dir=str(Path(self.base_config.output_dir) / "experiments"),
                name="multi_robot_vacuum",
            )
        experiment_dir = Path(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        RunSaver.save_json(asdict(self.base_config), experiment_dir / "experiment_config.json")

        seed_rows: List[Dict] = []
        for config in self.iter_condition_configs():
            condition_id = self._condition_id(config)
            run_dir = None
            if self.base_config.batch_config.save_per_seed_artifacts:
                run_dir = experiment_dir / condition_id / f"seed_{config.map_config.seed}"
            results, saved_dir = run_single_simulation(
                config,
                run_dir=run_dir,
                save_artifacts=self.base_config.batch_config.save_per_seed_artifacts,
            )
            row = self._result_row(config, results)
            if saved_dir is not None:
                row["run_dir"] = str(saved_dir)
            seed_rows.append(row)

        RunSaver.save_csv(seed_rows, experiment_dir / "seed_results.csv")
        aggregate = self.aggregate(seed_rows)
        RunSaver.save_json(aggregate, experiment_dir / "aggregate_summary.json")
        return experiment_dir

    def _condition_id(self, config: SimulationConfig) -> str:
        return (
            f"agents_{config.robot_config.num_agents}"
            f"__strategy_{config.coordination_config.strategy}"
            f"__sensor_{config.robot_config.sensor_range}"
            f"__obs_{config.map_config.obstacle_density:.2f}"
        )

    def _result_row(self, config: SimulationConfig, results: Dict) -> Dict:
        row = {
            "condition_id": self._condition_id(config),
            "seed": config.map_config.seed,
            "num_agents": config.robot_config.num_agents,
            "strategy": config.coordination_config.strategy,
            "sensor_range": config.robot_config.sensor_range,
            "obstacle_density": config.map_config.obstacle_density,
            "target_coverage": config.robot_config.target_coverage,
            "termination_reason": results.get("termination_reason"),
        }
        for metric in SUMMARY_METRICS:
            value = results.get(metric)
            # Keep CSV numeric-friendly; None means threshold was never reached.
            row[metric] = "" if value is None else value
        return row

    def aggregate(self, rows: List[Dict]) -> Dict:
        groups: Dict[str, List[Dict]] = {}
        for row in rows:
            groups.setdefault(row["condition_id"], []).append(row)

        summaries: List[Dict] = []
        for condition_id, condition_rows in sorted(groups.items()):
            first = condition_rows[0]
            summary: Dict = {
                "condition_id": condition_id,
                "num_agents": first["num_agents"],
                "strategy": first["strategy"],
                "sensor_range": first["sensor_range"],
                "obstacle_density": first["obstacle_density"],
                "n_runs": len(condition_rows),
            }
            for metric in SUMMARY_METRICS:
                values = [row[metric] for row in condition_rows if row.get(metric) != ""]
                if not values:
                    summary[f"{metric}_mean"] = None
                    summary[f"{metric}_std"] = None
                    continue
                numeric_values = [float(v) for v in values]
                summary[f"{metric}_mean"] = mean(numeric_values)
                summary[f"{metric}_std"] = stdev(numeric_values) if len(numeric_values) > 1 else 0.0
            summaries.append(summary)

        return {
            "n_conditions": len(summaries),
            "n_total_runs": len(rows),
            "summaries": summaries,
        }
