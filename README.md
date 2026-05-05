# Multi-Robot Cooperative Cleaning Simulation

A Python-based 2D grid simulation system for studying autonomous exploration, map building, path planning, cooperative cleaning, and resource competition among multiple robots in partially unknown indoor environments.

## Quick Start

```bash
# Run all experiments
python experiments/run_all_experiments.py

# Run individual experiments
python experiments/exp1_robot_count.py      # Robot count effect
python experiments/exp2_coordination_strategy.py  # Coordination strategies
python experiments/exp3_sensing.py           # Sensing capabilities
python experiments/exp4_charging_competition.py   # Charging competition
```

## Project Structure

```
├── environment/      # Grid map, obstacles, charging stations
├── agents/           # Robot state, behavior, battery logic
├── sensing/          # Sensor models (manhattan, euclidean, occluded)
├── mapping/          # Belief map, map fusion
├── exploration/      # Frontier detection
├── planning/         # A* path planning
├── control/          # Action execution
├── simulation/       # Simulation engine, coordination
├── metrics/          # Statistics collection
├── experiments/      # Experiment scripts
└── outputs/          # Results and charts
```

## Experiments

| # | Experiment | Variable |
|---|------------|----------|
| 1 | Robot Count | num_agents = [1, 2, 3, 4] |
| 2 | Coordination Strategy | independent, shared_map, goal_reservation, shared_map_reservation |
| 3 | Sensing | range = [2,3,4,5], modes = manhattan/euclidean/occluded_manhattan |
| 4 | Charging Competition | agents × station_capacity |

## Requirements

- Python 3.8+
- numpy, matplotlib (required for experiments)
