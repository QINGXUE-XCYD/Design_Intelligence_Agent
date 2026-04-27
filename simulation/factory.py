from __future__ import annotations

from agents.robot_agent import RobotAgent
from config.schema import SimulationConfig
from control.executor import ActionExecutor
from environment.dynamic_obstacles import RandomWalkDynamicObstaclePolicy
from environment.map_generator import MapGenerator
from exploration.frontier_detector import FrontierDetector
from mapping.occupancy_grid import OccupancyGrid, OccupancyState
from metrics.collector import MetricsCollector
from planning.astar_planner import AStarPlanner
from sensing.sensor_model import SensorModel
from simulation.engine import SimulationEngine


def build_simulation(config: SimulationConfig) -> SimulationEngine:
    """
    Build a simulation engine from a full configuration object.
    """
    map_generator = MapGenerator(config.map_config)
    env_map = map_generator.generate()
    starts = map_generator.sample_robot_starts(
        env_map,
        config.robot_config.num_agents,
    )

    # Robot start cells are also home charging stations.
    for start in starts:
        env_map.add_charging_station(start, station_type="home")

    # Optional extra public charging stations.
    public_chargers = map_generator.sample_charging_stations(
        env_map,
        config.map_config.additional_charging_stations,
        forbidden=starts,
    )
    for charger in public_chargers:
        env_map.add_charging_station(charger, station_type="public")

    agents = []
    for i, start_pos in enumerate(starts):
        belief_map = OccupancyGrid(
            width=config.map_config.width,
            height=config.map_config.height,
        )

        # Only the robot's own home charger is known initially.
        # Other home/public chargers are discovered by sensing or shared-map communication.
        belief_map.update_cell(start_pos, OccupancyState.FREE)

        sensor_seed = config.map_config.seed + 1009 * (i + 1)
        agent = RobotAgent(
            robot_id=i,
            start_pos=start_pos,
            belief_map=belief_map,
            sensor_model=SensorModel(
                config.robot_config.sensor_range,
                mode=config.robot_config.sensor_mode,
                false_positive_rate=config.robot_config.sensor_false_positive_rate,
                false_negative_rate=config.robot_config.sensor_false_negative_rate,
                seed=sensor_seed,
            ),
            frontier_detector=FrontierDetector(),
            planner=AStarPlanner(),
            executor=ActionExecutor(),
            enable_battery=config.robot_config.enable_battery,
            battery_capacity=config.robot_config.battery_capacity,
            low_battery_threshold=config.robot_config.low_battery_threshold,
            recharge_rate=config.robot_config.recharge_rate,
            battery_safety_margin=config.robot_config.battery_safety_margin,
            move_energy_cost=config.robot_config.move_energy_cost,
            sensing_energy_cost=config.robot_config.sensing_energy_cost,
            cleaning_energy_cost=config.robot_config.cleaning_energy_cost,
            communication_energy_cost=config.robot_config.communication_energy_cost,
        )
        agents.append(agent)

    dynamic_policy = None
    if config.map_config.dynamic_obstacle_count > 0:
        dynamic_policy = RandomWalkDynamicObstaclePolicy(
            count=config.map_config.dynamic_obstacle_count,
            seed=config.map_config.seed + 2027,
            move_probability=config.map_config.dynamic_obstacle_move_probability,
        )

    return SimulationEngine(
        env_map=env_map,
        agents=agents,
        metrics_collector=MetricsCollector(),
        max_steps=config.robot_config.max_steps,
        target_coverage=config.robot_config.target_coverage,
        coordination_strategy=config.coordination_config.strategy,
        communication_interval=config.coordination_config.communication_interval,
        communication_loss_rate=config.coordination_config.communication_loss_rate,
        charging_station_capacity=config.coordination_config.charging_station_capacity,
        dynamic_obstacle_policy=dynamic_policy,
        random_seed=config.map_config.seed + 4049,
    )
