from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from config.schema import SimulationConfig
from simulation.factory import build_simulation
from utils.run_saver import RunSaver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Experiment1Config:
    """
    Experiment 1: effect of robot count on cleaning efficiency.

    The main independent variable is the number of robots. Other parameters are
    kept fixed so the experiment answers a clean question suitable for the
    report.
    """

    output_dir: str = str(PROJECT_ROOT / "outputs" / "experiments")
    experiment_name: str = "exp1_robot_count"

    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    num_agents_values: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    coordination_strategy: str = "shared_map_reservation"
    sensor_range: int = 3
    obstacle_density: float = 0.15
    map_width: int = 30
    map_height: int = 30
    max_steps: int = 1500
    target_coverage: float = 0.95

    # Keep dynamic obstacles off for this first experiment.
    dynamic_obstacle_count: int = 0
    dynamic_obstacle_move_probability: float = 0.30

    # Battery/charging stays on so the experiment reflects the current full
    # system, but the parameters are fixed across conditions.
    enable_battery: bool = True
    charging_station_capacity: int = 1

    save_per_seed_artifacts: bool = False


def build_condition_config(base: Experiment1Config, num_agents: int, seed: int) -> SimulationConfig:
    config = SimulationConfig()
    config.output_dir = base.output_dir

    config.map_config.width = base.map_width
    config.map_config.height = base.map_height
    config.map_config.obstacle_density = base.obstacle_density
    config.map_config.seed = seed
    config.map_config.dynamic_obstacle_count = base.dynamic_obstacle_count
    config.map_config.dynamic_obstacle_move_probability = base.dynamic_obstacle_move_probability

    config.robot_config.num_agents = num_agents
    config.robot_config.sensor_range = base.sensor_range
    config.robot_config.max_steps = base.max_steps
    config.robot_config.target_coverage = base.target_coverage
    config.robot_config.enable_battery = base.enable_battery

    config.coordination_config.strategy = base.coordination_strategy
    config.coordination_config.charging_station_capacity = base.charging_station_capacity

    config.batch_config.enabled = False
    return config


def run_single_condition(
    experiment_dir: Path,
    base: Experiment1Config,
    num_agents: int,
    seed: int,
) -> Dict:
    config = build_condition_config(base, num_agents=num_agents, seed=seed)
    engine = build_simulation(config)
    results = engine.run()
    step_records = engine.metrics_collector.export_step_records()

    condition_id = f"agents_{num_agents}"
    seed_row = {
        "condition_id": condition_id,
        "seed": seed,
        "num_agents": num_agents,
        "strategy": base.coordination_strategy,
        "sensor_range": base.sensor_range,
        "obstacle_density": base.obstacle_density,
        "coverage_rate": results.get("coverage_rate"),
        "success": results.get("success"),
        "total_steps": results.get("total_steps"),
        "total_path_length": results.get("total_path_length"),
        "total_cleaned_cells": results.get("total_cleaned_cells"),
        "idle_steps": results.get("idle_steps"),
        "duplicate_visit_count": results.get("duplicate_visit_count"),
        "inter_agent_overlap_cells": results.get("inter_agent_overlap_cells"),
        "steps_to_80_coverage": results.get("steps_to_80_coverage"),
        "steps_to_90_coverage": results.get("steps_to_90_coverage"),
        "steps_to_95_coverage": results.get("steps_to_95_coverage"),
        "total_energy_used": results.get("total_energy_used"),
        "total_charge_wait_steps": results.get("total_charge_wait_steps"),
        "termination_reason": results.get("termination_reason"),
    }

    if base.save_per_seed_artifacts:
        run_dir = experiment_dir / condition_id / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        RunSaver.save_json(seed_row, run_dir / "metrics.json")
        RunSaver.save_json(step_records, run_dir / "step_records.json")

    return {
        "seed_row": seed_row,
        "step_records": step_records,
    }


def safe_mean(values: List[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: List[float]) -> float | None:
    if not values:
        return None
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_seed_rows(seed_rows: List[Dict], num_agents_values: List[int]) -> Dict:
    summaries: List[Dict] = []

    for num_agents in num_agents_values:
        rows = [row for row in seed_rows if row["num_agents"] == num_agents]
        if not rows:
            continue

        def numeric(metric: str) -> List[float]:
            return [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]

        success_values = [1.0 if row["success"] else 0.0 for row in rows]
        summary = {
            "condition_id": f"agents_{num_agents}",
            "num_agents": num_agents,
            "n_runs": len(rows),
            "strategy": rows[0]["strategy"],
            "sensor_range": rows[0]["sensor_range"],
            "obstacle_density": rows[0]["obstacle_density"],
            "coverage_rate_mean": safe_mean(numeric("coverage_rate")),
            "coverage_rate_std": safe_std(numeric("coverage_rate")),
            "success_rate": safe_mean(success_values),
            "total_steps_mean": safe_mean(numeric("total_steps")),
            "total_steps_std": safe_std(numeric("total_steps")),
            "total_path_length_mean": safe_mean(numeric("total_path_length")),
            "total_path_length_std": safe_std(numeric("total_path_length")),
            "duplicate_visit_count_mean": safe_mean(numeric("duplicate_visit_count")),
            "duplicate_visit_count_std": safe_std(numeric("duplicate_visit_count")),
            "inter_agent_overlap_cells_mean": safe_mean(numeric("inter_agent_overlap_cells")),
            "inter_agent_overlap_cells_std": safe_std(numeric("inter_agent_overlap_cells")),
            "steps_to_80_coverage_mean": safe_mean(numeric("steps_to_80_coverage")),
            "steps_to_80_coverage_std": safe_std(numeric("steps_to_80_coverage")),
            "steps_to_90_coverage_mean": safe_mean(numeric("steps_to_90_coverage")),
            "steps_to_90_coverage_std": safe_std(numeric("steps_to_90_coverage")),
            "steps_to_95_coverage_mean": safe_mean(numeric("steps_to_95_coverage")),
            "steps_to_95_coverage_std": safe_std(numeric("steps_to_95_coverage")),
            "total_energy_used_mean": safe_mean(numeric("total_energy_used")),
            "total_energy_used_std": safe_std(numeric("total_energy_used")),
            "total_charge_wait_steps_mean": safe_mean(numeric("total_charge_wait_steps")),
            "total_charge_wait_steps_std": safe_std(numeric("total_charge_wait_steps")),
        }
        summaries.append(summary)

    return {
        "n_conditions": len(summaries),
        "n_total_runs": len(seed_rows),
        "summaries": summaries,
    }


def build_mean_coverage_curves(
    coverage_by_condition: Dict[int, List[List[float]]],
    target_coverage: float,
) -> Dict[str, Dict]:
    """
    Align per-seed coverage curves by step index.

    When a run terminates earlier than others, its terminal coverage value is
    repeated so mean curves remain comparable across conditions.
    """
    curves: Dict[str, Dict] = {}

    for num_agents, runs in sorted(coverage_by_condition.items()):
        if not runs:
            continue

        max_len = max(len(run) for run in runs)
        aligned: List[List[float]] = []
        for run in runs:
            if not run:
                run = [0.0]
            padded = list(run)
            last_value = padded[-1]
            while len(padded) < max_len:
                padded.append(last_value)
            aligned.append(padded)

        mean_curve = [mean(step_values) for step_values in zip(*aligned)]
        curves[str(num_agents)] = {
            "num_agents": num_agents,
            "steps": list(range(max_len)),
            "mean_coverage_rate": mean_curve,
            "target_coverage": target_coverage,
        }

    return curves


def save_experiment_report(
    experiment_dir: Path,
    config: Experiment1Config,
    aggregate_summary: Dict,
) -> None:
    lines = [
        "# Experiment 1: Effect of Robot Count on Cleaning Efficiency",
        "",
        "## Independent Variable",
        "",
        f"- `num_agents in {config.num_agents_values}`",
        "",
        "## Controlled Variables",
        "",
        f"- `coordination_strategy = {config.coordination_strategy}`",
        f"- `sensor_range = {config.sensor_range}`",
        f"- `obstacle_density = {config.obstacle_density}`",
        f"- `map_size = {config.map_width} x {config.map_height}`",
        f"- `target_coverage = {config.target_coverage}`",
        f"- `dynamic_obstacle_count = {config.dynamic_obstacle_count}`",
        f"- `charging_station_capacity = {config.charging_station_capacity}`",
        "",
        "## Main Metrics",
        "",
        "- `coverage_rate`",
        "- `total_steps`",
        "- `total_path_length`",
        "- `duplicate_visit_count`",
        "- `inter_agent_overlap_cells`",
        "- `total_energy_used`",
        "- `total_charge_wait_steps`",
        "",
        "## Output Files",
        "",
        "- `seed_results.csv`: one row per seed",
        "- `aggregate_summary.json`: mean/std summary by robot count",
        "- `mean_coverage_curves.json`: mean coverage-vs-step curves",
        "- `coverage_curves_by_robot_count.png`: report-ready line chart",
        "- `mean_total_steps.png`: mean total steps by robot count",
        "- `mean_overlap.png`: mean overlap by robot count",
        "- `mean_energy.png`: mean energy use by robot count",
        "",
        "## Quick Summary",
        "",
    ]

    for summary in aggregate_summary.get("summaries", []):
        lines.append(
            "- "
            f"{summary['num_agents']} robots: "
            f"mean_steps={summary['total_steps_mean']:.2f}, "
            f"mean_coverage={summary['coverage_rate_mean']:.4f}, "
            f"mean_overlap={summary['inter_agent_overlap_cells_mean']:.2f}, "
            f"success_rate={summary['success_rate']:.2f}"
        )

    (experiment_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_plots(
    experiment_dir: Path,
    aggregate_summary: Dict,
    mean_curves: Dict[str, Dict],
    target_coverage: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Numeric outputs were saved, but plots were skipped.")
        return

    ordered = sorted(aggregate_summary["summaries"], key=lambda item: item["num_agents"])
    if not ordered:
        return

    # Plot 1: report-ready coverage-vs-step line chart.
    fig, ax = plt.subplots(figsize=(8, 5))
    for num_agents_text, curve in sorted(mean_curves.items(), key=lambda item: int(item[0])):
        ax.plot(curve["steps"], curve["mean_coverage_rate"], linewidth=2, label=f"{num_agents_text} robots")
    ax.axhline(target_coverage, color="gray", linestyle="--", linewidth=1.5, label="Target coverage")
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage Rate")
    ax.set_title("Coverage Curves by Robot Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(experiment_dir / "coverage_curves_by_robot_count.png", dpi=200)
    plt.close(fig)

    def bar_plot(metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str) -> None:
        x_values = [row["num_agents"] for row in ordered]
        y_values = [row[metric_mean] for row in ordered]
        errors = [0.0 if row[metric_std] is None else row[metric_std] for row in ordered]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x_values, y_values, yerr=errors, capsize=4)
        ax.set_xlabel("Number of Robots")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=200)
        plt.close(fig)

    bar_plot(
        metric_mean="total_steps_mean",
        metric_std="total_steps_std",
        ylabel="Mean Total Steps",
        title="Robot Count vs Mean Total Steps",
        filename="mean_total_steps.png",
    )
    bar_plot(
        metric_mean="inter_agent_overlap_cells_mean",
        metric_std="inter_agent_overlap_cells_std",
        ylabel="Mean Inter-Agent Overlap Cells",
        title="Robot Count vs Mean Overlap",
        filename="mean_overlap.png",
    )
    bar_plot(
        metric_mean="total_energy_used_mean",
        metric_std="total_energy_used_std",
        ylabel="Mean Total Energy Used",
        title="Robot Count vs Mean Energy Use",
        filename="mean_energy.png",
    )


def run_experiment(config: Experiment1Config | None = None) -> Path:
    config = config or Experiment1Config()
    experiment_dir = RunSaver.create_experiment_dir(
        base_dir=config.output_dir,
        name=config.experiment_name,
    )

    RunSaver.save_json(asdict(config), experiment_dir / "experiment_config.json")

    seed_rows: List[Dict] = []
    coverage_by_condition: Dict[int, List[List[float]]] = {n: [] for n in config.num_agents_values}

    for num_agents in config.num_agents_values:
        for seed in config.seeds:
            result = run_single_condition(
                experiment_dir=experiment_dir,
                base=config,
                num_agents=num_agents,
                seed=seed,
            )
            seed_rows.append(result["seed_row"])
            coverage_curve = [float(record["coverage_rate"]) for record in result["step_records"]]
            coverage_by_condition[num_agents].append(coverage_curve)

            print(
                f"[EXP1] agents={num_agents} seed={seed} "
                f"steps={result['seed_row']['total_steps']} "
                f"coverage={result['seed_row']['coverage_rate']:.4f} "
                f"success={result['seed_row']['success']}"
            )

    aggregate_summary = aggregate_seed_rows(seed_rows, config.num_agents_values)
    mean_curves = build_mean_coverage_curves(coverage_by_condition, config.target_coverage)

    RunSaver.save_csv(seed_rows, experiment_dir / "seed_results.csv")
    RunSaver.save_json(aggregate_summary, experiment_dir / "aggregate_summary.json")
    RunSaver.save_json(mean_curves, experiment_dir / "mean_coverage_curves.json")
    save_experiment_report(experiment_dir, config, aggregate_summary)
    generate_plots(experiment_dir, aggregate_summary, mean_curves, config.target_coverage)

    print(f"[EXP1] finished. Results saved to: {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    run_experiment()
