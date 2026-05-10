# Multi-Robot Cooperative Cleaning Simulation

A Python-based 2D grid simulation system for studying autonomous exploration, map building, path planning, cooperative cleaning, and resource competition among multiple robots in partially unknown indoor environments.

## Requirements

- Python 3.10
- matplotlib

### Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run all experiments
python main.py

# Run a specific experiment (1-4)
python main.py --exp 1   # Robot count effect
python main.py --exp 2   # Coordination strategies
python main.py --exp 3   # Sensing capabilities
python main.py --exp 4   # Charging competition
```

## Project Structure

```
├── main.py             # Entry point
├── environment/        # Grid map, obstacles, charging stations
├── agents/             # Robot state, behavior, battery logic
├── sensing/            # Sensor models (manhattan, euclidean, occluded)
├── mapping/            # Belief map, map fusion
├── exploration/        # Frontier detection
├── planning/           # A* path planning
├── control/            # Action execution
├── simulation/         # Simulation engine, coordination
├── metrics/            # Statistics collection
├── experiments/        # Experiment scripts
└── outputs/            # Results and charts
```

## Experiments

| # | Experiment | Variable |
|---|------------|----------|
| 1 | Robot Count | num_agents = [1, 2, 3, 4] |
| 2 | Coordination Strategy | independent, shared_map, goal_reservation, shared_map_reservation |
| 3 | Sensing | range = [2,3,4,5], modes = manhattan/euclidean/occluded_manhattan |
| 4 | Charging Competition | agents × station_capacity |


