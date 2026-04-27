from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from config.schema import SimulationConfig
from simulation.factory import build_simulation
from utils.run_saver import RunSaver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Experiment2Config:
    """
    Experiment 2: effect of coordination strategy on multi-robot performance.

    The independent variable is the coordination strategy. Robot count and the
    environment are held fixed so the experiment isolates collaboration effects.
    """

    output_dir: str = str(PROJECT_ROOT / "outputs" / "experiments")
    experiment_name: str = "exp2_coordination_strategy"

    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    strategies: List[str] = field(
        default_factory=lambda: [
            "independent",
            "shared_map",
            "goal_reservation",
            "shared_map_reservation",
        ]
    )

    num_agents: int = 3
    sensor_range: int = 3
    obstacle_density: float = 0.15
    map_width: int = 30
    map_height: int = 30
    max_steps: int = 1500
    target_coverage: float = 0.95

    # Keep dynamic obstacles off so the experiment focuses on coordination only.
    dynamic_obstacle_count: int = 0
    dynamic_obstacle_move_probability: float = 0.30

    enable_battery: bool = True
    charging_station_capacity: int = 1

    save_per_seed_artifacts: bool = False


def build_condition_config(base: Experiment2Config, strategy: str, seed: int) -> SimulationConfig:
    config = SimulationConfig()
    config.output_dir = base.output_dir

    config.map_config.width = base.map_width
    config.map_config.height = base.map_height
    config.map_config.obstacle_density = base.obstacle_density
    config.map_config.seed = seed
    config.map_config.dynamic_obstacle_count = base.dynamic_obstacle_count
    config.map_config.dynamic_obstacle_move_probability = base.dynamic_obstacle_move_probability

    config.robot_config.num_agents = base.num_agents
    config.robot_config.sensor_range = base.sensor_range
    config.robot_config.max_steps = base.max_steps
    config.robot_config.target_coverage = base.target_coverage
    config.robot_config.enable_battery = base.enable_battery

    config.coordination_config.strategy = strategy
    config.coordination_config.charging_station_capacity = base.charging_station_capacity

    config.batch_config.enabled = False
    return config


def run_single_condition(
    experiment_dir: Path,
    base: Experiment2Config,
    strategy: str,
    seed: int,
) -> Dict:
    config = build_condition_config(base, strategy=strategy, seed=seed)
    engine = build_simulation(config)
    results = engine.run()
    step_records = engine.metrics_collector.export_step_records()

    condition_id = strategy
    seed_row = {
        "condition_id": condition_id,
        "seed": seed,
        "strategy": strategy,
        "num_agents": base.num_agents,
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


def aggregate_seed_rows(seed_rows: List[Dict], strategies: List[str]) -> Dict:
    summaries: List[Dict] = []

    for strategy in strategies:
        rows = [row for row in seed_rows if row["strategy"] == strategy]
        if not rows:
            continue

        def numeric(metric: str) -> List[float]:
            return [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]

        success_values = [1.0 if row["success"] else 0.0 for row in rows]
        summary = {
            "condition_id": strategy,
            "strategy": strategy,
            "n_runs": len(rows),
            "num_agents": rows[0]["num_agents"],
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
    coverage_by_condition: Dict[str, List[List[float]]],
    target_coverage: float,
) -> Dict[str, Dict]:
    curves: Dict[str, Dict] = {}

    for strategy, runs in coverage_by_condition.items():
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
        curves[str(strategy)] = {
            "strategy": strategy,
            "steps": list(range(max_len)),
            "mean_coverage_rate": mean_curve,
            "target_coverage": target_coverage,
        }

    return curves


def save_experiment_report(
    experiment_dir: Path,
    config: Experiment2Config,
    aggregate_summary: Dict,
) -> None:
    lines = [
        "# Experiment 2: Effect of Coordination Strategy on Cleaning Efficiency",
        "",
        "## Independent Variable",
        "",
        f"- `strategy in {config.strategies}`",
        "",
        "## Controlled Variables",
        "",
        f"- `num_agents = {config.num_agents}`",
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
        "- `duplicate_visit_count`",
        "- `inter_agent_overlap_cells`",
        "- `success`",
        "- `total_energy_used`",
        "- `total_charge_wait_steps`",
        "",
        "## Output Files",
        "",
        "- `seed_results.csv`: one row per seed",
        "- `aggregate_summary.json`: mean/std summary by strategy",
        "- `mean_coverage_curves.json`: mean coverage-vs-step curves",
        "- `coverage_curves_by_strategy.png`: report-ready line chart",
        "- `mean_total_steps_by_strategy.png`: mean total steps by strategy",
        "- `mean_overlap_by_strategy.png`: mean overlap by strategy",
        "- `mean_duplicate_visits_by_strategy.png`: mean duplicate visits by strategy",
        "- `mean_energy_by_strategy.png`: mean energy use by strategy",
        "",
        "## Quick Summary",
        "",
    ]

    for summary in aggregate_summary.get("summaries", []):
        lines.append(
            "- "
            f"{summary['strategy']}: "
            f"mean_steps={summary['total_steps_mean']:.2f}, "
            f"mean_coverage={summary['coverage_rate_mean']:.4f}, "
            f"mean_overlap={summary['inter_agent_overlap_cells_mean']:.2f}, "
            f"mean_duplicate_visits={summary['duplicate_visit_count_mean']:.2f}, "
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

    ordered = list(aggregate_summary["summaries"])
    if not ordered:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for strategy, curve in mean_curves.items():
        ax.plot(curve["steps"], curve["mean_coverage_rate"], linewidth=2, label=strategy)
    ax.axhline(target_coverage, color="gray", linestyle="--", linewidth=1.5, label="Target coverage")
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage Rate")
    ax.set_title("Coverage Curves by Coordination Strategy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(experiment_dir / "coverage_curves_by_strategy.png", dpi=200)
    plt.close(fig)

    def bar_plot(metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str) -> None:
        x_labels = [row["strategy"] for row in ordered]
        y_values = [row[metric_mean] for row in ordered]
        errors = [0.0 if row[metric_std] is None else row[metric_std] for row in ordered]

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.bar(x_labels, y_values, yerr=errors, capsize=4)
        ax.set_xlabel("Coordination Strategy")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=200)
        plt.close(fig)

    bar_plot(
        metric_mean="total_steps_mean",
        metric_std="total_steps_std",
        ylabel="Mean Total Steps",
        title="Coordination Strategy vs Mean Total Steps",
        filename="mean_total_steps_by_strategy.png",
    )
    bar_plot(
        metric_mean="inter_agent_overlap_cells_mean",
        metric_std="inter_agent_overlap_cells_std",
        ylabel="Mean Inter-Agent Overlap Cells",
        title="Coordination Strategy vs Mean Overlap",
        filename="mean_overlap_by_strategy.png",
    )
    bar_plot(
        metric_mean="duplicate_visit_count_mean",
        metric_std="duplicate_visit_count_std",
        ylabel="Mean Duplicate Visit Count",
        title="Coordination Strategy vs Mean Duplicate Visits",
        filename="mean_duplicate_visits_by_strategy.png",
    )
    bar_plot(
        metric_mean="total_energy_used_mean",
        metric_std="total_energy_used_std",
        ylabel="Mean Total Energy Used",
        title="Coordination Strategy vs Mean Energy Use",
        filename="mean_energy_by_strategy.png",
    )


def run_experiment(config: Experiment2Config | None = None) -> Path:
    config = config or Experiment2Config()
    experiment_dir = RunSaver.create_experiment_dir(
        base_dir=config.output_dir,
        name=config.experiment_name,
    )

    RunSaver.save_json(asdict(config), experiment_dir / "experiment_config.json")

    seed_rows: List[Dict] = []
    coverage_by_condition: Dict[str, List[List[float]]] = {strategy: [] for strategy in config.strategies}

    for strategy in config.strategies:
        for seed in config.seeds:
            result = run_single_condition(
                experiment_dir=experiment_dir,
                base=config,
                strategy=strategy,
                seed=seed,
            )
            seed_rows.append(result["seed_row"])
            coverage_curve = [float(record["coverage_rate"]) for record in result["step_records"]]
            coverage_by_condition[strategy].append(coverage_curve)

            print(
                f"[EXP2] strategy={strategy} seed={seed} "
                f"steps={result['seed_row']['total_steps']} "
                f"coverage={result['seed_row']['coverage_rate']:.4f} "
                f"success={result['seed_row']['success']}"
            )

    aggregate_summary = aggregate_seed_rows(seed_rows, config.strategies)
    mean_curves = build_mean_coverage_curves(coverage_by_condition, config.target_coverage)

    RunSaver.save_csv(seed_rows, experiment_dir / "seed_results.csv")
    RunSaver.save_json(aggregate_summary, experiment_dir / "aggregate_summary.json")
    RunSaver.save_json(mean_curves, experiment_dir / "mean_coverage_curves.json")
    save_experiment_report(experiment_dir, config, aggregate_summary)
    generate_plots(experiment_dir, aggregate_summary, mean_curves, config.target_coverage)

    print(f"[EXP2] finished. Results saved to: {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    run_experiment()
