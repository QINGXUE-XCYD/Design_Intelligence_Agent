from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class RunSaver:
    """
    运行结果保存器 / Run artifact saver

    为每次仿真创建独立目录，并保存：
    - 指标结果
    - 逐步记录

    Creates a dedicated directory for each simulation run and saves:
    - summary metrics
    - per-step records
    """

    @staticmethod
    def create_run_dir(base_dir: str = "outputs") -> Path:
        """
        创建一次运行的输出目录 / Create output directory for one run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def save_metrics(metrics: Dict, save_path: str | Path) -> None:
        """
        保存汇总指标为 JSON / Save summary metrics as JSON
        """
        save_path = Path(save_path)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_step_records(step_records: List[Dict], save_path: str | Path) -> None:
        """
        保存逐步记录为 CSV / Save per-step records as CSV
        """
        save_path = Path(save_path)

        if not step_records:
            return

        fieldnames = list(step_records[0].keys())
        with save_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(step_records)