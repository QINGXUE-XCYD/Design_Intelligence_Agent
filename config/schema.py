from dataclasses import dataclass, field


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
    num_agents: int = 3
    sensor_range: int = 3
    max_steps: int = 500


@dataclass
class CoordinationConfig:
    """
    协同配置 / Coordination configuration
    定义固定角色、阶段释放和目标预留策略。
    Defines the fixed-role collaboration baseline.
    """
    scout_robot_id: int = 0
    cleaner_robot_ids: list[int] = field(default_factory=lambda: [1, 2])
    release_cleaners_after_scout: bool = True
    enable_goal_reservation: bool = True

@dataclass
class BatchConfig:
    """
    批量实验配置 / Batch experiment configuration
    """
    enabled: bool = False
    experiment_name: str = "fixed_role_three_robot_baseline"
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
    coordination_config: CoordinationConfig = field(default_factory=CoordinationConfig)
    batch_config: BatchConfig = field(default_factory=BatchConfig)
