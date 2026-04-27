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
class Experiment3Config:
    """
    Experiment 3: sensing capability and sensing mode.

    Part A studies sensor range.
    Part B compares different sensing modes, including an occlusion-aware model.
    """

    output_dir: str = str(PROJECT_ROOT / "outputs" / "experiments")
    experiment_name: str = "exp3_sensing"

    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55, 66, 77, 88, 99, 111])

    num_agents: int = 3
    coordination_strategy: str = "shared_map_reservation"
    obstacle_density: float = 0.15
    map_width: int = 30
    map_height: int = 30
    max_steps: int = 1500
    target_coverage: float = 0.95

    dynamic_obstacle_count: int = 0
    dynamic_obstacle_move_probability: float = 0.30

    enable_battery: bool = True
    charging_station_capacity: int = 1

    # Part A: range comparison under the default sensing mode.
    range_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    range_sensor_mode: str = "manhattan"

    # Part B: sensing-model comparison.
    mode_sensor_range: int = 3
    sensor_modes: List[str] = field(
        default_factory=lambda: [
            "manhattan",
            "euclidean",
            "occluded_manhattan",
        ]
    )

    save_per_seed_artifacts: bool = False


def build_base_simulation_config(base: Experiment3Config, seed: int) -> SimulationConfig:
    config = SimulationConfig()
    config.output_dir = base.output_dir

    config.map_config.width = base.map_width
    config.map_config.height = base.map_height
    config.map_config.obstacle_density = base.obstacle_density
    config.map_config.seed = seed
    config.map_config.dynamic_obstacle_count = base.dynamic_obstacle_count
    config.map_config.dynamic_obstacle_move_probability = base.dynamic_obstacle_move_probability

    config.robot_config.num_agents = base.num_agents
    config.robot_config.max_steps = base.max_steps
    config.robot_config.target_coverage = base.target_coverage
    config.robot_config.enable_battery = base.enable_battery

    config.coordination_config.strategy = base.coordination_strategy
    config.coordination_config.charging_station_capacity = base.charging_station_capacity

    config.batch_config.enabled = False
    return config


def build_range_config(base: Experiment3Config, sensor_range: int, seed: int) -> SimulationConfig:
    config = build_base_simulation_config(base, seed)
    config.robot_config.sensor_range = sensor_range
    config.robot_config.sensor_mode = base.range_sensor_mode
    config.robot_config.sensor_false_positive_rate = 0.0
    config.robot_config.sensor_false_negative_rate = 0.0
    return config


def build_mode_config(base: Experiment3Config, sensor_mode: str, seed: int) -> SimulationConfig:
    config = build_base_simulation_config(base, seed)
    config.robot_config.sensor_range = base.mode_sensor_range
    config.robot_config.sensor_mode = sensor_mode
    config.robot_config.sensor_false_positive_rate = 0.0
    config.robot_config.sensor_false_negative_rate = 0.0
    return config


def run_condition(
    experiment_dir: Path,
    category: str,
    condition_id: str,
    config: SimulationConfig,
    save_per_seed_artifacts: bool,
) -> Dict:
    engine = build_simulation(config)
    results = engine.run()
    step_records = engine.metrics_collector.export_step_records()

    seed_row = {
        "category": category,
        "condition_id": condition_id,
        "seed": config.map_config.seed,
        "num_agents": config.robot_config.num_agents,
        "strategy": config.coordination_config.strategy,
        "sensor_mode": config.robot_config.sensor_mode,
        "sensor_range": config.robot_config.sensor_range,
        "false_positive_rate": config.robot_config.sensor_false_positive_rate,
        "false_negative_rate": config.robot_config.sensor_false_negative_rate,
        "obstacle_density": config.map_config.obstacle_density,
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

    if save_per_seed_artifacts:
        run_dir = experiment_dir / category / condition_id / f"seed_{config.map_config.seed}"
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


def aggregate_rows(rows: List[Dict], condition_order: List[str]) -> Dict:
    summaries: List[Dict] = []
    for condition_id in condition_order:
        condition_rows = [row for row in rows if row["condition_id"] == condition_id]
        if not condition_rows:
            continue

        def numeric(metric: str) -> List[float]:
            return [float(row[metric]) for row in condition_rows if row.get(metric) not in ("", None)]

        success_values = [1.0 if row["success"] else 0.0 for row in condition_rows]
        first = condition_rows[0]
        summaries.append(
            {
                "condition_id": condition_id,
                "category": first["category"],
                "n_runs": len(condition_rows),
                "sensor_mode": first["sensor_mode"],
                "sensor_range": first["sensor_range"],
                "false_positive_rate": first["false_positive_rate"],
                "false_negative_rate": first["false_negative_rate"],
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
        )

    return {
        "n_conditions": len(summaries),
        "n_total_runs": len(rows),
        "summaries": summaries,
    }


def build_mean_coverage_curves(coverage_by_condition: Dict[str, List[List[float]]]) -> Dict[str, Dict]:
    curves: Dict[str, Dict] = {}
    for condition_id, runs in coverage_by_condition.items():
        if not runs:
            continue
        max_len = max(len(run) for run in runs)
        aligned: List[List[float]] = []
        for run in runs:
            padded = list(run or [0.0])
            last_value = padded[-1]
            while len(padded) < max_len:
                padded.append(last_value)
            aligned.append(padded)
        curves[condition_id] = {
            "condition_id": condition_id,
            "steps": list(range(max_len)),
            "mean_coverage_rate": [mean(step_values) for step_values in zip(*aligned)],
        }
    return curves


def save_section_report(
    section_dir: Path,
    title: str,
    config_lines: List[str],
    metric_lines: List[str],
    output_lines: List[str],
    aggregate_summary: Dict,
) -> None:
    lines = [f"# {title}", "", "## 实验设置", ""]
    lines.extend(config_lines)
    lines.extend(["", "## 主要指标", ""])
    lines.extend(metric_lines)
    lines.extend(["", "## 输出文件", ""])
    lines.extend(output_lines)
    lines.extend(["", "## 快速结果总结", ""])
    for summary in aggregate_summary.get("summaries", []):
        lines.append(
            "- "
            f"{summary['condition_id']}: "
            f"mean_steps={summary['total_steps_mean']:.2f}, "
            f"mean_coverage={summary['coverage_rate_mean']:.4f}, "
            f"mean_duplicate_visits={summary['duplicate_visit_count_mean']:.2f}, "
            f"success_rate={summary['success_rate']:.2f}"
        )
    (section_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_plots(
    section_dir: Path,
    aggregate_summary: Dict,
    mean_curves: Dict[str, Dict],
    x_labels: List[str],
    x_title: str,
    figure_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"[WARN] matplotlib is not installed. Plots were skipped for {figure_prefix}.")
        return

    ordered = [next(item for item in aggregate_summary["summaries"] if item["condition_id"] == label) for label in x_labels]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for condition_id in x_labels:
        curve = mean_curves.get(condition_id)
        if curve is None:
            continue
        ax.plot(curve["steps"], curve["mean_coverage_rate"], linewidth=2, label=condition_id)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage Rate")
    ax.set_title(f"Coverage Curves by {x_title}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(section_dir / f"{figure_prefix}_coverage_curves.png", dpi=200)
    plt.close(fig)

    def bar_plot(metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str) -> None:
        y_values = [row[metric_mean] for row in ordered]
        errors = [0.0 if row[metric_std] is None else row[metric_std] for row in ordered]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.bar(x_labels, y_values, yerr=errors, capsize=4)
        ax.set_xlabel(x_title)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(section_dir / filename, dpi=200)
        plt.close(fig)

    bar_plot(
        "total_steps_mean",
        "total_steps_std",
        "Mean Total Steps",
        f"{x_title} vs Mean Total Steps",
        f"{figure_prefix}_mean_total_steps.png",
    )
    bar_plot(
        "duplicate_visit_count_mean",
        "duplicate_visit_count_std",
        "Mean Duplicate Visit Count",
        f"{x_title} vs Mean Duplicate Visits",
        f"{figure_prefix}_mean_duplicate_visits.png",
    )
    bar_plot(
        "total_energy_used_mean",
        "total_energy_used_std",
        "Mean Total Energy Used",
        f"{x_title} vs Mean Energy Use",
        f"{figure_prefix}_mean_energy.png",
    )


def run_experiment(config: Experiment3Config | None = None) -> Path:
    config = config or Experiment3Config()
    experiment_dir = RunSaver.create_experiment_dir(
        base_dir=config.output_dir,
        name=config.experiment_name,
    )
    RunSaver.save_json(asdict(config), experiment_dir / "experiment_config.json")

    # Part A1: sensor range comparison.
    range_dir = experiment_dir / "part_a_sensor_range"
    range_dir.mkdir(parents=True, exist_ok=True)
    range_rows: List[Dict] = []
    range_condition_order = [f"range_{value}" for value in config.range_values]
    range_curves: Dict[str, List[List[float]]] = {cid: [] for cid in range_condition_order}

    for sensor_range in config.range_values:
        condition_id = f"range_{sensor_range}"
        for seed in config.seeds:
            sim_config = build_range_config(config, sensor_range=sensor_range, seed=seed)
            result = run_condition(
                experiment_dir=experiment_dir,
                category="part_a_sensor_range",
                condition_id=condition_id,
                config=sim_config,
                save_per_seed_artifacts=config.save_per_seed_artifacts,
            )
            range_rows.append(result["seed_row"])
            range_curves[condition_id].append([float(record["coverage_rate"]) for record in result["step_records"]])
            print(
                f"[EXP3-RANGE] range={sensor_range} seed={seed} "
                f"steps={result['seed_row']['total_steps']} "
                f"coverage={result['seed_row']['coverage_rate']:.4f} "
                f"success={result['seed_row']['success']}"
            )

    range_summary = aggregate_rows(range_rows, range_condition_order)
    range_mean_curves = build_mean_coverage_curves(range_curves)
    RunSaver.save_csv(range_rows, range_dir / "seed_results.csv")
    RunSaver.save_json(range_summary, range_dir / "aggregate_summary.json")
    RunSaver.save_json(range_mean_curves, range_dir / "mean_coverage_curves.json")
    save_section_report(
        range_dir,
        title="Experiment 3A-1: Sensor Range",
        config_lines=[
            f"- `sensor_range in {config.range_values}`",
            f"- `sensor_mode = {config.range_sensor_mode}`",
            f"- `num_agents = {config.num_agents}`",
            f"- `coordination_strategy = {config.coordination_strategy}`",
            f"- `obstacle_density = {config.obstacle_density}`",
            f"- `map_size = {config.map_width} x {config.map_height}`",
        ],
        metric_lines=[
            "- `coverage_rate`",
            "- `total_steps`",
            "- `duplicate_visit_count`",
            "- `steps_to_80_coverage`",
            "- `steps_to_90_coverage`",
            "- `total_energy_used`",
        ],
        output_lines=[
            "- `seed_results.csv`",
            "- `aggregate_summary.json`",
            "- `mean_coverage_curves.json`",
            "- `range_coverage_curves.png`",
            "- `range_mean_total_steps.png`",
            "- `range_mean_duplicate_visits.png`",
            "- `range_mean_energy.png`",
        ],
        aggregate_summary=range_summary,
    )
    generate_plots(
        range_dir,
        range_summary,
        range_mean_curves,
        range_condition_order,
        "Sensor Range",
        "range",
    )

    # Part B: sensing mode comparison.
    mode_dir = experiment_dir / "part_b_sensor_modes"
    mode_dir.mkdir(parents=True, exist_ok=True)
    mode_rows: List[Dict] = []
    mode_condition_order = list(config.sensor_modes)
    mode_curves: Dict[str, List[List[float]]] = {cid: [] for cid in mode_condition_order}

    for sensor_mode in config.sensor_modes:
        for seed in config.seeds:
            sim_config = build_mode_config(config, sensor_mode=sensor_mode, seed=seed)
            result = run_condition(
                experiment_dir=experiment_dir,
                category="part_b_sensor_modes",
                condition_id=sensor_mode,
                config=sim_config,
                save_per_seed_artifacts=config.save_per_seed_artifacts,
            )
            mode_rows.append(result["seed_row"])
            mode_curves[sensor_mode].append([float(record["coverage_rate"]) for record in result["step_records"]])
            print(
                f"[EXP3-MODE] mode={sensor_mode} seed={seed} "
                f"steps={result['seed_row']['total_steps']} "
                f"coverage={result['seed_row']['coverage_rate']:.4f} "
                f"success={result['seed_row']['success']}"
            )

    mode_summary = aggregate_rows(mode_rows, mode_condition_order)
    mode_mean_curves = build_mean_coverage_curves(mode_curves)
    RunSaver.save_csv(mode_rows, mode_dir / "seed_results.csv")
    RunSaver.save_json(mode_summary, mode_dir / "aggregate_summary.json")
    RunSaver.save_json(mode_mean_curves, mode_dir / "mean_coverage_curves.json")
    save_section_report(
        mode_dir,
        title="Experiment 3B: Sensor Mode Comparison",
        config_lines=[
            f"- `sensor_modes = {config.sensor_modes}`",
            f"- `sensor_range = {config.mode_sensor_range}`",
            f"- `num_agents = {config.num_agents}`",
            f"- `coordination_strategy = {config.coordination_strategy}`",
            "- `noise = 0.0 / 0.0`",
        ],
        metric_lines=[
            "- `coverage_rate`",
            "- `total_steps`",
            "- `duplicate_visit_count`",
            "- `steps_to_80_coverage`",
            "- `steps_to_90_coverage`",
            "- `total_energy_used`",
        ],
        output_lines=[
            "- `seed_results.csv`",
            "- `aggregate_summary.json`",
            "- `mean_coverage_curves.json`",
            "- `mode_coverage_curves.png`",
            "- `mode_mean_total_steps.png`",
            "- `mode_mean_duplicate_visits.png`",
            "- `mode_mean_energy.png`",
        ],
        aggregate_summary=mode_summary,
    )
    generate_plots(
        mode_dir,
        mode_summary,
        mode_mean_curves,
        mode_condition_order,
        "Sensor Mode",
        "mode",
    )

    print(f"[EXP3] finished. Results saved to: {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    run_experiment()
