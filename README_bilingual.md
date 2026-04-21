# Design and Evaluation of Multi-Robot Cooperative Vacuum Cleaning in Partially Unknown Dynamic Indoor Environments

# 面向部分未知动态室内环境的多扫地机器人协同清扫系统设计与实验评估

---

## 1. Project Overview | 项目概述

**English**

This project is a modular Python simulation framework for studying autonomous vacuum-cleaning robots in a 2D grid-based indoor environment. It is designed for coursework in **Intelligent Agents / Designing Intelligent Agents**, with emphasis on environment modelling, autonomous decision making, repeatable experiments, performance evaluation, and future extensibility.

The current codebase already supports the core pipeline of:
- map generation
- robot sensing
- belief map updating
- frontier-based exploration
- A* path planning
- cleaning / coverage tracking
- simulation result saving
- visualisation and experiment output

The implementation is intentionally structured for later extension to multi-robot collaboration, shared maps, task allocation, communication constraints, battery logic, and other advanced mechanisms.

**中文**

本项目是一个基于 Python 的模块化仿真框架，用于研究二维栅格室内环境中的自主扫地机器人行为。项目面向 **Intelligent Agents / Designing Intelligent Agents** 相关 coursework，重点强调环境建模、自主决策、可重复实验、性能评估以及后续可扩展性。

当前代码已经支持以下核心流程：
- 地图生成
- 机器人感知
- belief map 更新
- 基于 frontier 的探索
- A* 路径规划
- 清扫与覆盖率统计
- 仿真结果保存
- 可视化与实验输出

整体实现特意采用可扩展结构，便于后续继续加入多机器人协作、共享地图、任务分配、通信约束、电量机制以及其他更复杂的模块。

---

## 2. Project Goals | 项目目标

**English**

The long-term objective of this project is to study how one or more autonomous cleaning robots can efficiently explore, map, and clean a partially unknown indoor environment with obstacles.

Representative research questions include:
1. How does the number of robots affect cleaning efficiency?
2. Does communication or shared mapping improve collaboration performance?
3. How do different coordination or task allocation strategies affect coverage, overlap, and makespan?

**中文**

本项目的长期目标是研究一个或多个自主清扫机器人如何在存在障碍物、且部分未知的室内环境中高效地完成探索、建图与清扫任务。

代表性研究问题包括：
1. 机器人数量变化会如何影响清扫效率？
2. 通信或共享地图是否能够提升协作效果？
3. 不同协同或任务分配策略会如何影响覆盖率、重复率和完成时间？

---

## 3. Current Implementation Status | 当前实现状态

### Implemented | 已实现

**English**
- 2D grid-based indoor map representation
- Static obstacle generation
- Robot state management
- Local sensing and observation update
- Occupancy-grid belief map
- Frontier-based exploration baseline
- A* planner
- Step-based simulation engine
- Coverage metrics collection
- Overview rendering and coverage curve plotting
- Single-run artifact saving
- Batch experiment output structure

**中文**
- 二维栅格室内地图表示
- 静态障碍生成
- 机器人状态管理
- 局部感知与观测更新
- occupancy-grid belief map
- 基于 frontier 的探索基线方法
- A* 路径规划器
- 按 step 推进的仿真引擎
- 覆盖率指标采集
- 总览图与覆盖率曲线绘制
- 单次运行结果保存
- 批量实验输出结构

### Planned / In Progress | 计划中 / 开发中

**English**
- True multi-robot coordination
- Shared belief maps / map fusion
- Goal reservation and task allocation
- Dynamic obstacle policies
- Communication delay / packet loss modelling
- Battery and charging extensions
- Sensor noise
- More advanced experimental baselines

**中文**
- 真正的多机器人协同逻辑
- 共享 belief map / 地图融合
- 目标预留与任务分配
- 动态障碍策略
- 通信延迟 / 丢包建模
- 电量与充电扩展
- 传感器噪声
- 更高级的实验基线方法

---

## 4. Project Structure | 项目目录结构

```text
DIA/
│  .gitignore
│  main.py
│
├─agents/
│  │  agent_state.py
│  │  robot_agent.py
│
├─config/
│  │  schema.py
│
├─control/
│  │  action.py
│  │  executor.py
│
├─environment/
│  │  cell.py
│  │  grid_map.py
│  │  map_generator.py
│
├─exploration/
│  │  frontier_detector.py
│
├─mapping/
│  │  occupancy_grid.py
│
├─metrics/
│  │  collector.py
│
├─outputs/
│  ├─experiments/
│  │  └─single_agent_baseline_.../
│  │          aggregate_summary.json
│  │          experiment_config.json
│  │          seed_results.csv
│  │
│  └─run_.../
│          agent_step_log.csv
│          agent_trajectories.json
│          coverage_curve.png
│          metrics.json
│          per_agent_summary.json
│          run_metadata.json
│          run_overview.png
│          step_records.csv
│
├─planning/
│  │  astar_planner.py
│  │  planner_base.py
│
├─sensing/
│  │  observation.py
│  │  sensor_model.py
│
├─simulation/
│  │  engine.py
│
├─utils/
│  │  run_saver.py
│
└─visualisation/
   │  renderer.py
```

---

## 5. Module Description | 模块说明

### `environment/`
**English**: Defines the ground-truth environment, including grid cells, map structure, and map generation.

**中文**：定义真实环境，包括栅格单元、地图结构以及地图生成逻辑。

### `agents/`
**English**: Contains robot runtime state and high-level agent behaviour, such as goals, paths, modes, and trajectory tracking.

**中文**：包含机器人运行时状态以及高层行为逻辑，例如目标、路径、模式和轨迹记录。

### `sensing/`
**English**: Handles local sensing and observation generation around the robot.

**中文**：负责机器人周围局部感知与观测生成。

### `mapping/`
**English**: Stores the robot’s internal occupancy-grid belief map and cleaned-state information.

**中文**：存储机器人内部的 occupancy-grid belief map 以及已清扫状态信息。

### `exploration/`
**English**: Provides frontier detection used during exploration.

**中文**：提供探索阶段使用的 frontier 检测逻辑。

### `planning/`
**English**: Contains planners, currently including an A* baseline.

**中文**：包含路径规划模块，目前实现了 A* 基线方法。

### `control/`
**English**: Converts high-level plans into local executable actions.

**中文**：将高层规划结果转为局部可执行动作。

### `metrics/`
**English**: Collects simulation statistics such as coverage and path length.

**中文**：采集覆盖率、路径长度等仿真指标。

### `simulation/`
**English**: Runs the step-based simulation loop and decides termination.

**中文**：运行按 step 推进的仿真循环，并负责结束判定。

### `utils/`
**English**: Saves run artifacts and experiment outputs.

**中文**：负责保存运行结果与实验输出文件。

### `visualisation/`
**English**: Generates the run overview figure and coverage curve.

**中文**：生成运行总览图和覆盖率曲线。

---

## 6. Core Pipeline | 核心运行流程

**English**

In each simulation step, the robot pipeline is conceptually:
1. sense the local environment
2. update the belief map
3. choose an exploration or cleaning target
4. plan a path
5. execute an action
6. clean the current cell if appropriate
7. record metrics and logs

**中文**

在每一个仿真 step 中，机器人的核心流程可以概括为：
1. 感知局部环境
2. 更新 belief map
3. 选择探索或补扫目标
4. 规划路径
5. 执行动作
6. 在合适时机清扫当前位置
7. 记录指标与日志

---

## 7. Requirements | 环境依赖

**English**

The project is implemented in Python. Based on the current codebase, the main external libraries are:
- `numpy`
- `matplotlib`

You may install them with:

```bash
pip install numpy matplotlib
```

**中文**

项目基于 Python 实现。根据当前代码结构，主要外部依赖包括：
- `numpy`
- `matplotlib`

可使用以下命令安装：

```bash
pip install numpy matplotlib
```

---

## 8. How to Run | 运行方式

### Single Run | 单次运行

**English**

Run the main entry file from the project root:

```bash
python main.py
```

This will execute one simulation and save outputs into a timestamped directory under `outputs/`.

**中文**

在项目根目录运行：

```bash
python main.py
```

程序会执行一次仿真，并将结果保存到 `outputs/` 下带时间戳的目录中。

### Batch Experiments | 批量实验

**English**

If batch mode is enabled in the configuration, the program will run multiple seeds and save experiment-level outputs under `outputs/experiments/`.

To active batch mode, set `config/schema.BatchConfig.Enabled = True`

Typical batch outputs include:
- `experiment_config.json`
- `seed_results.csv`
- `aggregate_summary.json`

**中文**

如果在配置中启用了 batch 模式，程序会按多个 seed 执行实验，并将实验级结果保存到 `outputs/experiments/` 下。

激活Batch Mode， 修改`config/schema.BatchConfig.Enabled = True`

典型批量输出包括：
- `experiment_config.json`
- `seed_results.csv`
- `aggregate_summary.json`

---

## 9. Output Files | 输出文件说明

### Single Run Outputs | 单次运行输出

**English**
- `run_metadata.json`: run ID, timestamp, seed, config snapshot, start/final positions
- `metrics.json`: final summary metrics
- `per_agent_summary.json`: per-agent summary statistics
- `agent_trajectories.json`: stored robot trajectories and phase information
- `step_records.csv`: global step-level metrics
- `agent_step_log.csv`: per-step, per-agent position and state logs
- `run_overview.png`: run overview figure
- `coverage_curve.png`: coverage progression over time

**中文**
- `run_metadata.json`：运行编号、时间戳、seed、配置快照、起点与终点信息
- `metrics.json`：最终汇总指标
- `per_agent_summary.json`：逐机器人统计信息
- `agent_trajectories.json`：机器人轨迹与阶段信息
- `step_records.csv`：全局 step 级指标记录
- `agent_step_log.csv`：逐 step、逐机器人的位置与状态日志
- `run_overview.png`：运行总览图
- `coverage_curve.png`：覆盖率随时间变化曲线

### Batch Outputs | 批量实验输出

**English**
- `experiment_config.json`: experiment-level configuration
- `seed_results.csv`: one row per seed
- `aggregate_summary.json`: mean, standard deviation, success rate, and other aggregate statistics

**中文**
- `experiment_config.json`：实验级配置
- `seed_results.csv`：每个 seed 一行结果
- `aggregate_summary.json`：均值、标准差、成功率等聚合统计

---

## 10. Visualisation | 可视化说明

**English**

The renderer currently produces:
- a run overview figure combining maps, trajectories, and belief maps
- a coverage curve showing cleaning progress over simulation steps

These outputs are useful for debugging, qualitative analysis, and coursework reporting.

**中文**

当前可视化模块能够生成：
- 结合地图、轨迹和 belief map 的运行总览图
- 展示覆盖率随 step 变化的曲线图

这些结果适合用于调试、定性分析和 coursework 报告展示。

---

## 11. Current Limitations | 当前限制

**English**

At the current stage, the project is still a baseline framework. Some limitations remain:
- multi-robot coordination is not yet fully completed
- shared-map communication is not yet fully implemented
- dynamic obstacles are not yet the main focus of the current baseline
- relative map alignment between independent robots remains future work
- advanced task allocation strategies are still to be added

**中文**

在当前阶段，本项目仍然属于基线框架，仍存在一些限制：
- 多机器人协同逻辑尚未完全完成
- 共享地图通信尚未完全实现
- 动态障碍还不是当前基线的主要部分
- 多机器人独立地图之间的相对对齐仍是后续工作
- 更高级的任务分配策略尚待加入

---

## 12. Future Work | 后续扩展方向

**English**

Planned future extensions include:
- shared/global belief maps
- goal reservation strategies
- region allocation or auction-based coordination
- battery and charging station modelling
- communication delay / loss
- sensor noise
- dynamic obstacle interaction
- more rigorous experiment benchmarking for coursework evaluation

**中文**

计划中的扩展方向包括：
- 共享 / 全局 belief map
- 目标预留策略
- 区域划分或拍卖式协同方法
- 电量与充电站建模
- 通信延迟 / 丢包
- 传感器噪声
- 动态障碍交互机制
- 面向 coursework 的更系统 benchmark 实验

---

## 13. Suggested Use in Coursework | 在 Coursework 中的建议用法

**English**

This repository can support the following coursework workflow:
1. run baseline experiments with fixed seeds
2. compare parameter settings or strategies
3. inspect qualitative run figures
4. report quantitative averages and standard deviations
5. discuss limitations and future improvements

**中文**

这个仓库可以支持如下 coursework 工作流：
1. 使用固定 seed 运行基线实验
2. 比较不同参数设置或策略
3. 查看定性运行图像结果
4. 汇报定量平均值与标准差
5. 讨论系统限制与未来改进方向

---

## 14. Entry Point | 程序入口

**English**

The main entry point of the project is:

```text
main.py
```

**中文**

项目当前的主要运行入口为：

```text
main.py
```

---

## 15. Notes | 备注

**English**

This project is under active development. The architecture is intentionally modular so that additional functionality can be integrated incrementally without rewriting the whole system.

**中文**

本项目仍在持续迭代中。整体架构刻意保持模块化，方便后续在不推翻整体结构的前提下逐步加入更多功能。
