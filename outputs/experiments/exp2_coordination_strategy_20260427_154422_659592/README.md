# Experiment 2: Effect of Coordination Strategy on Cleaning Efficiency

## Independent Variable

- `strategy in ['independent', 'shared_map', 'goal_reservation', 'shared_map_reservation']`

## Controlled Variables

- `num_agents = 3`
- `sensor_range = 3`
- `obstacle_density = 0.15`
- `map_size = 30 x 30`
- `target_coverage = 0.95`
- `dynamic_obstacle_count = 0`
- `charging_station_capacity = 1`

## Main Metrics

- `coverage_rate`
- `total_steps`
- `duplicate_visit_count`
- `inter_agent_overlap_cells`
- `success`
- `total_energy_used`
- `total_charge_wait_steps`

## Output Files

- `seed_results.csv`: one row per seed
- `aggregate_summary.json`: mean/std summary by strategy
- `mean_coverage_curves.json`: mean coverage-vs-step curves
- `coverage_curves_by_strategy.png`: report-ready line chart
- `mean_total_steps_by_strategy.png`: mean total steps by strategy
- `mean_overlap_by_strategy.png`: mean overlap by strategy
- `mean_duplicate_visits_by_strategy.png`: mean duplicate visits by strategy
- `mean_energy_by_strategy.png`: mean energy use by strategy

## Quick Summary

- independent: mean_steps=771.90, mean_coverage=0.9505, mean_overlap=525.80, mean_duplicate_visits=1471.40, success_rate=1.00
- shared_map: mean_steps=543.60, mean_coverage=0.9510, mean_overlap=261.30, mean_duplicate_visits=810.10, success_rate=1.00
- goal_reservation: mean_steps=764.00, mean_coverage=0.9505, mean_overlap=515.60, mean_duplicate_visits=1448.70, success_rate=1.00
- shared_map_reservation: mean_steps=483.20, mean_coverage=0.9510, mean_overlap=223.20, mean_duplicate_visits=650.90, success_rate=1.00