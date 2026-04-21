from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MapConfig:
    """
    地图配置 / Map configuration
    用于定义二维栅格环境的基本参数。
    Used to define the basic parameters of a 2D grid environment.
    """
    width: int = 20
    height: int = 20
    obstacle_density: float = 0.15
    seed: int = 42


@dataclass
class RobotConfig:
    """
    机器人配置 / Robot configuration
    定义机器人数量、传感器范围和基础运动参数。
    Defines robot count, sensor range, and basic motion parameters.
    """
    num_agents: int = 1
    sensor_range: int = 3
    max_steps: int = 500

@dataclass
class BatchConfig:
    """
    批量实验配置 / Batch experiment configuration
    """
    enabled: bool = True
    experiment_name: str = "single_agent_baseline"
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62, 72, 82])
    base_output_dir: str = "outputs/experiments"
    save_per_seed_details: bool = False


@dataclass
class SimulationConfig:
    """
    仿真总配置 / Overall simulation configuration
    聚合地图、机器人和仿真运行参数。
    Aggregates map, robot, and simulation runtime settings.
    """
    map_config: MapConfig = field(default_factory=MapConfig)
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    batch_config: BatchConfig = field(default_factory=BatchConfig)