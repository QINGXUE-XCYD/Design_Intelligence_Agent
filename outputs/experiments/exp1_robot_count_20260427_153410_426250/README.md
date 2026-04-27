# Experiment 1: Effect of Robot Count on Cleaning Efficiency

## Independent Variable

- `num_agents in [1, 2, 3, 4]`

## Controlled Variables

- `coordination_strategy = shared_map_reservation`
- `sensor_range = 3`
- `obstacle_density = 0.15`
- `map_size = 30 x 30`
- `target_coverage = 0.95`
- `dynamic_obstacle_count = 0`
- `charging_station_capacity = 1`

## Main Metrics

- `coverage_rate`
- `total_steps`
- `total_path_length`
- `duplicate_visit_count`
- `inter_agent_overlap_cells`
- `total_energy_used`
- `total_charge_wait_steps`

## Output Files

- `seed_results.csv`: one row per seed
- `aggregate_summary.json`: mean/std summary by robot count
- `mean_coverage_curves.json`: mean coverage-vs-step curves
- `coverage_curves_by_robot_count.png`: report-ready line chart
- `mean_total_steps.png`: mean total steps by robot count
- `mean_overlap.png`: mean overlap by robot count
- `mean_energy.png`: mean energy use by robot count

## Quick Summary

- 1 robots: mean_steps=1390.20, mean_coverage=0.9189, mean_overlap=0.00, success_rate=0.50
- 2 robots: mean_steps=700.10, mean_coverage=0.9508, mean_overlap=180.70, success_rate=1.00
- 3 robots: mean_steps=483.20, mean_coverage=0.9510, mean_overlap=223.20, success_rate=1.00
- 4 robots: mean_steps=353.70, mean_coverage=0.9508, mean_overlap=231.10, success_rate=1.00