from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from agents.robot_agent import RobotAgent
from environment.grid_map import GridMap
from mapping.occupancy_grid import OccupancyState


class MapRenderer:
    """
    地图渲染器 / Map renderer

    总览图布局：
    - 第一行：初始地图 + 每个机器人的轨迹图
    - 第二行：共享 belief 叠加图 + 每个机器人的 belief map

    Overview layout:
    - Row 1: initial map + one trajectory panel per robot
    - Row 2: shared belief overlay + one belief-map panel per robot
    """

    def render_run_overview(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        save_path: str | Path,
    ) -> None:
        """
        保存总览图 / Save the run overview figure

        布局：
        2 rows x (1 + number_of_agents) columns
        """
        n_panels = 1 + len(agents)

        fig, axes = plt.subplots(
            2,
            n_panels,
            figsize=(6 * n_panels, 12),
            squeeze=False,
        )

        # =========================
        # 第一行 / Row 1
        # =========================

        # (0, 0): Initial Map
        initial_canvas = self._build_env_canvas(env_map, show_cleaned=False)
        self._draw_background(
            ax=axes[0, 0],
            canvas=initial_canvas,
            env_map=env_map,
            title="Initial Map",
        )
        for idx, agent in enumerate(agents):
            color = self._get_agent_color(idx)
            sx, sy = agent.state.trajectory[0]
            axes[0, 0].scatter(
                sx,
                sy,
                marker="o",
                s=80,
                color=color,
                label=f"Robot {agent.state.robot_id} start",
            )
        axes[0, 0].legend(loc="upper right")

        # (0, i): Trajectory per robot
        for idx, agent in enumerate(agents, start=1):
            final_canvas = self._build_env_canvas(env_map, show_cleaned=True)
            self._draw_background(
                ax=axes[0, idx],
                canvas=final_canvas,
                env_map=env_map,
                title=f"Robot {agent.state.robot_id} Trajectory",
            )
            self._draw_agent_trajectory(
                ax=axes[0, idx],
                agent=agent,
                color=self._get_agent_color(idx - 1),
            )
            axes[0, idx].legend(loc="upper right")

        # =========================
        # 第二行 / Row 2
        # =========================

        # (1, 0): Shared Belief Overlay
        shared_belief_canvas = self._build_shared_belief_canvas(agents, env_map)
        self._draw_background(
            ax=axes[1, 0],
            canvas=shared_belief_canvas,
            env_map=env_map,
            title="Shared Belief Map (Overlay)",
        )

        for idx, agent in enumerate(agents):
            color = self._get_agent_color(idx)
            cx, cy = agent.state.position
            axes[1, 0].scatter(
                cx,
                cy,
                marker="o",
                s=70,
                color=color,
                label=f"Robot {agent.state.robot_id}",
            )
        axes[1, 0].legend(loc="upper right")

        # (1, i): Individual Belief Map
        for idx, agent in enumerate(agents, start=1):
            color = self._get_agent_color(idx - 1)
            belief_canvas = self._build_single_belief_canvas(
                agent=agent,
                color=color,
                env_map=env_map,
            )

            self._draw_background(
                ax=axes[1, idx],
                canvas=belief_canvas,
                env_map=env_map,
                title=f"Robot {agent.state.robot_id} Belief Map",
            )

            cx, cy = agent.state.position
            axes[1, idx].scatter(
                cx,
                cy,
                marker="o",
                s=70,
                color=color,
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    def plot_coverage_curve(
        self,
        step_records: List[dict],
        save_path: str | Path,
        title: str = "Coverage Rate Over Time",
    ) -> None:
        """
        保存覆盖率曲线 / Save coverage-rate curve
        """
        if not step_records:
            return

        steps = [r["step"] for r in step_records]
        coverage = [r["coverage_rate"] for r in step_records]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, coverage, linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Coverage Rate")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    # =========================================================
    # 基础环境图 / Environment map canvas
    # =========================================================

    def _build_env_canvas(self, env_map: GridMap, show_cleaned: bool) -> np.ndarray:
        """
        构造环境地图画布 / Build RGB canvas for environment map
        """
        canvas = np.ones((env_map.height, env_map.width, 3), dtype=float)

        for x in range(env_map.width):
            for y in range(env_map.height):
                pos = (x, y)

                if pos in env_map.static_obstacles:
                    canvas[y, x] = [0.0, 0.0, 0.0]
                elif pos in env_map.dynamic_obstacles:
                    canvas[y, x] = [1.0, 0.6, 0.0]
                elif show_cleaned and pos in env_map.cleaned_cells:
                    canvas[y, x] = [0.80, 0.92, 1.00]
                else:
                    canvas[y, x] = [1.0, 1.0, 1.0]

        return canvas

    # =========================================================
    # belief map 可视化 / Belief map visualisation
    # =========================================================

    def _build_single_belief_canvas(
        self,
        agent: RobotAgent,
        color: str,
        env_map: GridMap,
    ) -> np.ndarray:
        """
        构造单个机器人的 belief map 画布
        Build a canvas for one robot's belief map
        """
        canvas = np.ones((env_map.height, env_map.width, 3), dtype=float)
        rgb = np.array(mcolors.to_rgb(color), dtype=float)

        unknown_color = np.array([0.95, 0.95, 0.95], dtype=float)
        free_tint = 0.55 * np.ones(3) + 0.45 * rgb
        occupied_color = np.array([0.0, 0.0, 0.0], dtype=float)

        for x in range(env_map.width):
            for y in range(env_map.height):
                state = self._get_belief_state(agent, (x, y))

                if state == OccupancyState.UNKNOWN:
                    canvas[y, x] = unknown_color
                elif state == OccupancyState.FREE:
                    canvas[y, x] = free_tint
                elif state == OccupancyState.OCCUPIED:
                    canvas[y, x] = occupied_color
                else:
                    canvas[y, x] = unknown_color

        return canvas

    def _build_shared_belief_canvas(
        self,
        agents: List[RobotAgent],
        env_map: GridMap,
    ) -> np.ndarray:
        """
        构造所有机器人 belief map 的叠加图
        Build an overlay canvas for all robots' belief maps

        规则：
        - 任何机器人认为 OCCUPIED -> 黑色
        - 一个或多个机器人认为 FREE -> 颜色平均混合
        - 全 UNKNOWN -> 浅灰
        """
        canvas = np.ones((env_map.height, env_map.width, 3), dtype=float)
        unknown_color = np.array([0.95, 0.95, 0.95], dtype=float)
        occupied_color = np.array([0.0, 0.0, 0.0], dtype=float)

        robot_colors = [
            np.array(mcolors.to_rgb(self._get_agent_color(i)), dtype=float)
            for i in range(len(agents))
        ]

        for x in range(env_map.width):
            for y in range(env_map.height):
                free_colors = []
                has_occupied = False

                for i, agent in enumerate(agents):
                    state = self._get_belief_state(agent, (x, y))

                    if state == OccupancyState.OCCUPIED:
                        has_occupied = True
                    elif state == OccupancyState.FREE:
                        free_colors.append(robot_colors[i])

                if has_occupied:
                    canvas[y, x] = occupied_color
                elif free_colors:
                    mean_color = np.mean(np.stack(free_colors, axis=0), axis=0)
                    # 和白色混合一下，保持浅色 / Blend with white for a softer overlay
                    canvas[y, x] = 0.50 * np.ones(3) + 0.50 * mean_color
                else:
                    canvas[y, x] = unknown_color

        return canvas

    def _get_belief_state(self, agent: RobotAgent, pos: Tuple[int, int]) -> OccupancyState:
        """
        读取某个机器人 belief map 上某格的状态
        Read one cell state from a robot belief map
        """
        return agent.belief_map.get_cell(pos)

    # =========================================================
    # 绘图辅助 / Plot helpers
    # =========================================================

    def _draw_background(self, ax, canvas: np.ndarray, env_map: GridMap, title: str) -> None:
        """
        绘制地图背景 / Draw map background
        """
        ax.imshow(canvas, origin="lower")
        ax.set_title(title)
        ax.set_xlim(-0.5, env_map.width - 0.5)
        ax.set_ylim(-0.5, env_map.height - 0.5)
        ax.set_xticks(range(env_map.width))
        ax.set_yticks(range(env_map.height))
        ax.grid(True, which="both", color="lightgray", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")

    def _draw_agent_trajectory(self, ax, agent: RobotAgent, color: str) -> None:
        """
        绘制单个机器人的轨迹 / Draw trajectory of one robot
        """
        traj = agent.state.trajectory
        if not traj:
            return

        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        ax.plot(xs, ys, linewidth=2, alpha=0.9, color=color, label=f"Robot {agent.state.robot_id}")

        sx, sy = traj[0]
        ex, ey = traj[-1]
        ax.scatter(sx, sy, marker="o", s=70, color=color)
        ax.scatter(ex, ey, marker="x", s=90, color=color)

    def _get_agent_color(self, idx: int) -> str:
        """
        获取机器人颜色 / Get agent color from matplotlib cycle
        """
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return color_cycle[idx % len(color_cycle)]