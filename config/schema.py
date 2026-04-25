from dataclasses import dataclass, field
from typing import List


@dataclass
class MapConfig:
    """
    地图配置 / Map configuration
    """
    width: int = 30
    height: int = 30
    obstacle_density: float = 0.15
    seed: int = 42

    # Optional public charging stations in addition to robot start stations.
    additional_charging_stations: int = 2

    # Optional dynamic-obstacle extension. Disabled by default.
    dynamic_obstacle_count: int = 0
    dynamic_obstacle_move_probability: float = 0.30


@dataclass
class RobotConfig:
    """
    机器人配置 / Robot configuration
    """
    num_agents: int = 2
    sensor_range: int = 3
    max_steps: int = 1500
    target_coverage: float = 0.95

    # Optional noisy-sensing extension. Disabled by default.
    sensor_false_positive_rate: float = 0.0
    sensor_false_negative_rate: float = 0.0

    # Battery-aware charging extension. Disabled by default.
    enable_battery: bool = True
    battery_capacity: float = 80.0
    low_battery_threshold: float = 22.0
    recharge_rate: float = 15.0

    # New battery-aware planning parameters.
    battery_safety_margin: float = 15.0
    move_energy_cost: float = 1.0
    sensing_energy_cost: float = 0.05
    cleaning_energy_cost: float = 0.15
    communication_energy_cost: float = 0.05


@dataclass
class CoordinationConfig:
    """
    多机器人协同配置 / Multi-robot coordination configuration.

    Supported strategies:
    - independent: each robot uses only its own belief map
    - shared_map: robots periodically fuse local belief maps
    - goal_reservation: robots avoid selecting goals already assigned to peers
    - shared_map_reservation: shared map + goal reservation
    """
    # strategy: str = "independent"
    strategy: str = "shared_map_reservation"
    communication_interval: int = 1
    communication_loss_rate: float = 0.0

    # Charging station service capacity. A value of 1 means only one robot can
    # actively recharge at the same station in the same simulation step.
    charging_station_capacity: int = 1


@dataclass
class BatchConfig:
    """
    批量实验配置 / Batch-experiment configuration.
    """
    enabled: bool = False
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    num_agents_values: List[int] = field(default_factory=lambda: [1, 2, 3])
    strategies: List[str] = field(
        default_factory=lambda: ["independent", "shared_map", "shared_map_reservation"]
    )
    sensor_ranges: List[int] = field(default_factory=lambda: [3])
    obstacle_densities: List[float] = field(default_factory=lambda: [0.15])
    save_per_seed_artifacts: bool = False


@dataclass
class SimulationConfig:
    """
    仿真总配置 / Overall simulation configuration.
    """
    map_config: MapConfig = field(default_factory=MapConfig)
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    coordination_config: CoordinationConfig = field(default_factory=CoordinationConfig)
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    output_dir: str = "outputs"
