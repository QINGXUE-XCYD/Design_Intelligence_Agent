"""
Multi-Robot Cooperative Cleaning Simulation - Entry Point

Usage:
    python main.py          # Run all experiments
    python main.py --exp 1  # Run experiment 1 only
    python main.py --exp 2  # Run experiment 2 only
    python main.py --exp 3  # Run experiment 3 only
    python main.py --exp 4  # Run experiment 4 only
"""
from __future__ import annotations

import argparse
import sys

EXPERIMENTS = {
    1: ("Experiment 1: Robot Count Effect", "experiments.exp1_robot_count"),
    2: ("Experiment 2: Coordination Strategy", "experiments.exp2_coordination_strategy"),
    3: ("Experiment 3: Sensing Capabilities", "experiments.exp3_sensing"),
    4: ("Experiment 4: Charging Competition", "experiments.exp4_charging_competition"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-robot cleaning experiments")
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a specific experiment (1-4). If omitted, all experiments are run.",
    )
    args = parser.parse_args()

    if args.exp is not None:
        targets = [args.exp]
    else:
        targets = list(EXPERIMENTS.keys())

    for exp_num in targets:
        label, module_path = EXPERIMENTS[exp_num]
        print(f"\n[RUN] {label} ...")

        import importlib
        module = importlib.import_module(module_path)
        output_dir = module.run_experiment()
        print(f"[DONE] {label} -> {output_dir}")

    print("\n[RUN] All selected experiments completed.")


if __name__ == "__main__":
    main()
