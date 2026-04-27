# Experiment 3B: Sensor Mode Comparison

## 实验设置

- `sensor_modes = ['manhattan', 'euclidean', 'occluded_manhattan']`
- `sensor_range = 3`
- `num_agents = 3`
- `coordination_strategy = shared_map_reservation`
- `noise = 0.0 / 0.0`

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
- `mode_coverage_curves.png`
- `mode_mean_total_steps.png`
- `mode_mean_duplicate_visits.png`
- `mode_mean_energy.png`

## 快速结果总结

- manhattan: mean_steps=483.20, mean_coverage=0.9510, mean_duplicate_visits=650.90, success_rate=1.00
- euclidean: mean_steps=478.80, mean_coverage=0.9505, mean_duplicate_visits=631.40, success_rate=1.00
- occluded_manhattan: mean_steps=471.10, mean_coverage=0.9508, mean_duplicate_visits=628.10, success_rate=1.00