# Experiment 3A-1: Sensor Range

## 实验设置

- `sensor_range in [2, 3, 4, 5]`
- `sensor_mode = manhattan`
- `num_agents = 3`
- `coordination_strategy = shared_map_reservation`
- `obstacle_density = 0.15`
- `map_size = 30 x 30`

## 主要指标

- `coverage_rate`
- `total_steps`
- `duplicate_visit_count`
- `steps_to_80_coverage`
- `steps_to_90_coverage`
- `total_energy_used`

## 输出文件

- `seed_results.csv`
- `aggregate_summary.json`
- `mean_coverage_curves.json`
- `range_coverage_curves.png`
- `range_mean_total_steps.png`
- `range_mean_duplicate_visits.png`
- `range_mean_energy.png`

## 快速结果总结

- range_2: mean_steps=529.40, mean_coverage=0.9507, mean_duplicate_visits=764.70, success_rate=1.00
- range_3: mean_steps=483.20, mean_coverage=0.9510, mean_duplicate_visits=650.90, success_rate=1.00
- range_4: mean_steps=444.80, mean_coverage=0.9510, mean_duplicate_visits=559.40, success_rate=1.00
- range_5: mean_steps=459.10, mean_coverage=0.9507, mean_duplicate_visits=567.80, success_rate=1.00