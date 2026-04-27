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
class Experiment4Config:
    """
    Experiment 4: charging competition under battery stress.

    This experiment is intentionally configured to trigger charging more often
    than the default system settings, so the effect of charger contention is
    visible in the results.
    """

    output_dir: str = str(PROJECT_ROOT / "outputs" / "experiments")
    experiment_name: str = "exp4_charging_competition"

    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55, 66, 77, 88, 99, 111])

    # Two-dimensional design: robot count x charging-station capacity.
    num_agents_values: List[int] = field(default_factory=lambda: [2, 3, 4])
    charging_station_capacity_values: List[int] = field(default_factory=lambda: [1, 2])

    coordination_strategy: str = "shared_map_reservation"
    sensor_range: int = 3
    sensor_mode: str = "manhattan"
    obstacle_density: float = 0.15
    map_width: int = 30
    map_height: int = 30
    max_steps: int = 2000
    target_coverage: float = 0.95

    dynamic_obstacle_count: int = 0
    dynamic_obstacle_move_probability: float = 0.30

    # Keep the battery enabled and make it more restrictive than the default
    # settings so charging is triggered more frequently.
    enable_battery: bool = True
    battery_capacity: float = 100.0
    low_battery_threshold: float = 22.0
    recharge_rate: float = 12.0
    battery_safety_margin: float = 15.0

    # Remove extra public chargers to make charger contention easier to observe.
    additional_charging_stations: int = 0

    save_per_seed_artifacts: bool = False


def build_condition_config(
    base: Experiment4Config,
    num_agents: int,
    charging_station_capacity: int,
    seed: int,
) -> SimulationConfig:
    config = SimulationConfig()
    config.output_dir = base.output_dir

    config.map_config.width = base.map_width
    config.map_config.height = base.map_height
    config.map_config.obstacle_density = base.obstacle_density
    config.map_config.seed = seed
    config.map_config.dynamic_obstacle_count = base.dynamic_obstacle_count
    config.map_config.dynamic_obstacle_move_probability = base.dynamic_obstacle_move_probability
    config.map_config.additional_charging_stations = base.additional_charging_stations

    config.robot_config.num_agents = num_agents
    config.robot_config.sensor_range = base.sensor_range
    config.robot_config.sensor_mode = base.sensor_mode
    config.robot_config.max_steps = base.max_steps
    config.robot_config.target_coverage = base.target_coverage
    config.robot_config.enable_battery = base.enable_battery
    config.robot_config.battery_capacity = base.battery_capacity
    config.robot_config.low_battery_threshold = base.low_battery_threshold
    config.robot_config.recharge_rate = base.recharge_rate
    config.robot_config.battery_safety_margin = base.battery_safety_margin

    config.coordination_config.strategy = base.coordination_strategy
    config.coordination_config.charging_station_capacity = charging_station_capacity

    config.batch_config.enabled = False
    return config


def run_single_condition(
    experiment_dir: Path,
    base: Experiment4Config,
    num_agents: int,
    charging_station_capacity: int,
    seed: int,
) -> Dict:
    config = build_condition_config(
        base=base,
        num_agents=num_agents,
        charging_station_capacity=charging_station_capacity,
        seed=seed,
    )
    engine = build_simulation(config)
    results = engine.run()

    condition_id = f"agents_{num_agents}__capacity_{charging_station_capacity}"
    seed_row = {
        "condition_id": condition_id,
        "seed": seed,
        "num_agents": num_agents,
        "charging_station_capacity": charging_station_capacity,
        "strategy": base.coordination_strategy,
        "sensor_range": base.sensor_range,
        "sensor_mode": base.sensor_mode,
        "obstacle_density": base.obstacle_density,
        "battery_capacity": base.battery_capacity,
        "low_battery_threshold": base.low_battery_threshold,
        "recharge_rate": base.recharge_rate,
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
        "total_charging_steps": results.get("total_charging_steps"),
        "total_charging_events": results.get("total_charging_events"),
        "total_charge_wait_steps": results.get("total_charge_wait_steps"),
        "total_low_battery_returns": results.get("total_low_battery_returns"),
        "total_battery_budget_returns": results.get("total_battery_budget_returns"),
        "total_battery_depletion_count": results.get("total_battery_depletion_count"),
        "termination_reason": results.get("termination_reason"),
    }

    if base.save_per_seed_artifacts:
        run_dir = experiment_dir / condition_id / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        RunSaver.save_json(seed_row, run_dir / "metrics.json")

    return {"seed_row": seed_row}


def safe_mean(values: List[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: List[float]) -> float | None:
    if not values:
        return None
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_seed_rows(seed_rows: List[Dict], condition_order: List[str]) -> Dict:
    summaries: List[Dict] = []

    for condition_id in condition_order:
        rows = [row for row in seed_rows if row["condition_id"] == condition_id]
        if not rows:
            continue

        def numeric(metric: str) -> List[float]:
            return [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]

        success_values = [1.0 if row["success"] else 0.0 for row in rows]
        first = rows[0]
        summaries.append(
            {
                "condition_id": condition_id,
                "num_agents": first["num_agents"],
                "charging_station_capacity": first["charging_station_capacity"],
                "n_runs": len(rows),
                "success_rate": safe_mean(success_values),
                "coverage_rate_mean": safe_mean(numeric("coverage_rate")),
                "coverage_rate_std": safe_std(numeric("coverage_rate")),
                "total_steps_mean": safe_mean(numeric("total_steps")),
                "total_steps_std": safe_std(numeric("total_steps")),
                "total_energy_used_mean": safe_mean(numeric("total_energy_used")),
                "total_energy_used_std": safe_std(numeric("total_energy_used")),
                "total_charging_steps_mean": safe_mean(numeric("total_charging_steps")),
                "total_charging_steps_std": safe_std(numeric("total_charging_steps")),
                "total_charging_events_mean": safe_mean(numeric("total_charging_events")),
                "total_charging_events_std": safe_std(numeric("total_charging_events")),
                "total_charge_wait_steps_mean": safe_mean(numeric("total_charge_wait_steps")),
                "total_charge_wait_steps_std": safe_std(numeric("total_charge_wait_steps")),
                "total_low_battery_returns_mean": safe_mean(numeric("total_low_battery_returns")),
                "total_low_battery_returns_std": safe_std(numeric("total_low_battery_returns")),
                "total_battery_budget_returns_mean": safe_mean(numeric("total_battery_budget_returns")),
                "total_battery_budget_returns_std": safe_std(numeric("total_battery_budget_returns")),
                "total_battery_depletion_count_mean": safe_mean(numeric("total_battery_depletion_count")),
                "total_battery_depletion_count_std": safe_std(numeric("total_battery_depletion_count")),
            }
        )

    return {
        "n_conditions": len(summaries),
        "n_total_runs": len(seed_rows),
        "summaries": summaries,
    }


def save_experiment_report(experiment_dir: Path, config: Experiment4Config, aggregate_summary: Dict) -> None:
    lines = [
        "# Experiment 4: Charging Competition Under Battery Stress",
        "",
        "## Independent Variables",
        "",
        f"- `num_agents in {config.num_agents_values}`",
        f"- `charging_station_capacity in {config.charging_station_capacity_values}`",
        "",
        "## Controlled Variables",
        "",
        f"- `coordination_strategy = {config.coordination_strategy}`",
        f"- `sensor_range = {config.sensor_range}`",
        f"- `sensor_mode = {config.sensor_mode}`",
        f"- `obstacle_density = {config.obstacle_density}`",
        f"- `map_size = {config.map_width} x {config.map_height}`",
        f"- `target_coverage = {config.target_coverage}`",
        f"- `dynamic_obstacle_count = {config.dynamic_obstacle_count}`",
        "",
        "## Battery Stress Settings",
        "",
        f"- `battery_capacity = {config.battery_capacity}`",
        f"- `low_battery_threshold = {config.low_battery_threshold}`",
        f"- `recharge_rate = {config.recharge_rate}`",
        f"- `battery_safety_margin = {config.battery_safety_margin}`",
        f"- `additional_charging_stations = {config.additional_charging_stations}`",
        "",
        "## Main Metrics",
        "",
        "- `coverage_rate`",
        "- `total_steps`",
        "- `total_energy_used`",
        "- `total_charging_steps`",
        "- `total_charging_events`",
        "- `total_charge_wait_steps`",
        "- `total_low_battery_returns`",
        "- `total_battery_budget_returns`",
        "",
        "## Output Files",
        "",
        "- `seed_results.csv`",
        "- `aggregate_summary.json`",
        "- `mean_charge_wait_heatmap.png`",
        "- `mean_total_steps_heatmap.png`",
        "- `success_rate_heatmap.png`",
        "- `mean_charging_events_heatmap.png`",
        "",
        "## Quick Summary",
        "",
    ]

    for summary in aggregate_summary.get("summaries", []):
        lines.append(
            "- "
            f"{summary['condition_id']}: "
            f"mean_steps={summary['total_steps_mean']:.2f}, "
            f"mean_wait={summary['total_charge_wait_steps_mean']:.2f}, "
            f"mean_events={summary['total_charging_events_mean']:.2f}, "
            f"success_rate={summary['success_rate']:.2f}"
        )

    (experiment_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_matrix(
    aggregate_summary: Dict,
    num_agents_values: List[int],
    charging_station_capacity_values: List[int],
    metric: str,
) -> List[List[float]]:
    matrix: List[List[float]] = []
    for num_agents in num_agents_values:
        row: List[float] = []
        for capacity in charging_station_capacity_values:
            summary = next(
                item
                for item in aggregate_summary["summaries"]
                if item["num_agents"] == num_agents and item["charging_station_capacity"] == capacity
            )
            value = summary.get(metric)
            row.append(0.0 if value is None else float(value))
        matrix.append(row)
    return matrix


def generate_heatmaps(
    experiment_dir: Path,
    aggregate_summary: Dict,
    num_agents_values: List[int],
    charging_station_capacity_values: List[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Numeric outputs were saved, but plots were skipped.")
        return

    def draw_heatmap(metric: str, title: str, filename: str, fmt: str = ".1f") -> None:
        matrix = build_matrix(
            aggregate_summary=aggregate_summary,
            num_agents_values=num_agents_values,
            charging_station_capacity_values=charging_station_capacity_values,
            metric=metric,
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(charging_station_capacity_values)))
        ax.set_xticklabels(charging_station_capacity_values)
        ax.set_yticks(range(len(num_agents_values)))
        ax.set_yticklabels(num_agents_values)
        ax.set_xlabel("Charging Station Capacity")
        ax.set_ylabel("Number of Robots")
        ax.set_title(title)

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, format(value, fmt), ha="center", va="center", color="white" if value > max(map(max, matrix)) * 0.5 else "black")

        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=200)
        plt.close(fig)

    draw_heatmap(
        metric="total_charge_wait_steps_mean",
        title="Mean Charge-Wait Steps",
        filename="mean_charge_wait_heatmap.png",
    )
    draw_heatmap(
        metric="total_steps_mean",
        title="Mean Total Steps",
        filename="mean_total_steps_heatmap.png",
    )
    draw_heatmap(
        metric="success_rate",
        title="Success Rate",
        filename="success_rate_heatmap.png",
        fmt=".2f",
    )
    draw_heatmap(
        metric="total_charging_events_mean",
        title="Mean Charging Events",
        filename="mean_charging_events_heatmap.png",
    )


def run_experiment(config: Experiment4Config | None = None) -> Path:
    config = config or Experiment4Config()
    experiment_dir = RunSaver.create_experiment_dir(
        base_dir=config.output_dir,
        name=config.experiment_name,
    )
    RunSaver.save_json(asdict(config), experiment_dir / "experiment_config.json")

    seed_rows: List[Dict] = []
    condition_order: List[str] = []

    for num_agents in config.num_agents_values:
        for charging_station_capacity in config.charging_station_capacity_values:
            condition_id = f"agents_{num_agents}__capacity_{charging_station_capacity}"
            condition_order.append(condition_id)

            for seed in config.seeds:
                result = run_single_condition(
                    experiment_dir=experiment_dir,
                    base=config,
                    num_agents=num_agents,
                    charging_station_capacity=charging_station_capacity,
                    seed=seed,
                )
                seed_rows.append(result["seed_row"])
                print(
                    f"[EXP4] agents={num_agents} capacity={charging_station_capacity} seed={seed} "
                    f"steps={result['seed_row']['total_steps']} "
                    f"wait={result['seed_row']['total_charge_wait_steps']} "
                    f"events={result['seed_row']['total_charging_events']} "
                    f"success={result['seed_row']['success']}"
                )

    aggregate_summary = aggregate_seed_rows(seed_rows, condition_order)
    RunSaver.save_csv(seed_rows, experiment_dir / "seed_results.csv")
    RunSaver.save_json(aggregate_summary, experiment_dir / "aggregate_summary.json")
    save_experiment_report(experiment_dir, config, aggregate_summary)
    generate_heatmaps(
        experiment_dir,
        aggregate_summary,
        config.num_agents_values,
        config.charging_station_capacity_values,
    )

    print(f"[EXP4] finished. Results saved to: {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    run_experiment()
