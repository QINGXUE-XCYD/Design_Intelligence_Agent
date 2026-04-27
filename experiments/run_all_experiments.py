from __future__ import annotations

from experiments.exp1_robot_count import run_experiment as run_exp1
from experiments.exp2_coordination_strategy import run_experiment as run_exp2
from experiments.exp3_sensing import run_experiment as run_exp3
from experiments.exp4_charging_competition import run_experiment as run_exp4


def main() -> None:
    experiment_runs = [
        ("Experiment 1", run_exp1),
        ("Experiment 2", run_exp2),
        ("Experiment 3", run_exp3),
        ("Experiment 4", run_exp4),
    ]

    for label, runner in experiment_runs:
        print(f"[RUN-ALL] starting {label}...")
        output_dir = runner()
        print(f"[RUN-ALL] finished {label}: {output_dir}")

    print("[RUN-ALL] all experiments completed.")


if __name__ == "__main__":
    main()
