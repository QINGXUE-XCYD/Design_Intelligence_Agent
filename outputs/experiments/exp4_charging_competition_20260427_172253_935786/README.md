# Experiment 4: Charging Competition Under Battery Stress

## Independent Variables

- `num_agents in [2, 3, 4]`
- `charging_station_capacity in [1, 2]`

## Controlled Variables

- `coordination_strategy = shared_map_reservation`
- `sensor_range = 3`
- `sensor_mode = manhattan`
- `obstacle_density = 0.15`
- `map_size = 30 x 30`
- `target_coverage = 0.95`
- `dynamic_obstacle_count = 0`

## Battery Stress Settings

- `battery_capacity = 100.0`
- `low_battery_threshold = 22.0`
- `recharge_rate = 12.0`
- `battery_safety_margin = 15.0`
- `additional_charging_stations = 0`

## Main Metrics

- `coverage_rate`
- `total_steps`
- `total_energy_used`
- `total_charging_steps`
- `total_charging_events`
- `total_charge_wait_steps`
- `total_low_battery_returns`
- `total_battery_budget_returns`

## Output Files

- `seed_results.csv`
- `aggregate_summary.json`
- `mean_charge_wait_heatmap.png`
- `mean_total_steps_heatmap.png`
- `success_rate_heatmap.png`
- `mean_charging_events_heatmap.png`

## Quick Summary

- agents_2__capacity_1: mean_steps=790.70, mean_wait=71.70, mean_events=79.10, success_rate=1.00
- agents_2__capacity_2: mean_steps=748.50, mean_wait=0.00, mean_events=19.50, success_rate=1.00
- agents_3__capacity_1: mean_steps=486.70, mean_wait=107.70, mean_events=81.00, success_rate=1.00
- agents_3__capacity_2: mean_steps=451.00, mean_wait=16.10, mean_events=29.80, success_rate=1.00
- agents_4__capacity_1: mean_steps=354.10, mean_wait=104.30, mean_events=74.60, success_rate=1.00
- agents_4__capacity_2: mean_steps=330.90, mean_wait=26.30, mean_events=37.20, success_rate=1.00