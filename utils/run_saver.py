from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class RunSaver:
    """
    运行结果保存器 / Run artifact saver.
    """

    @staticmethod
    def create_run_dir(base_dir: str = "outputs", prefix: str = "run") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def create_experiment_dir(base_dir: str = "outputs/experiments", name: str = "experiment") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        exp_dir = Path(base_dir) / f"{name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    @staticmethod
    def save_json(data, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_metrics(metrics: Dict, save_path: str | Path) -> None:
        RunSaver.save_json(metrics, save_path)

    @staticmethod
    def save_csv(records: List[Dict], save_path: str | Path, fieldnames: Optional[List[str]] = None) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not records and not fieldnames:
            return
        if fieldnames is None:
            fieldnames = sorted({key for record in records for key in record.keys()})
        with save_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def save_step_records(step_records: List[Dict], save_path: str | Path) -> None:
        RunSaver.save_csv(step_records, save_path)

    @staticmethod
    def save_agent_step_records(agent_step_records: List[Dict], save_path: str | Path) -> None:
        RunSaver.save_csv(agent_step_records, save_path)
