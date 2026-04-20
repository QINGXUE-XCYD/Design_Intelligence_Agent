from __future__ import annotations
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from agents.robot_agent import RobotAgent
from environment.grid_map import GridMap


class MapRenderer:
    """
    地图渲染器 / Map renderer

    统一输出一张 overview 画布：
    - 第 1 张：初始地图
    - 后续每张：一个机器人的轨迹图

    Render one overview canvas:
    - Panel 1: initial map
    - Remaining panels: one trajectory panel per robot
    """

    def render_run_overview(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        save_path: str | Path,
    ) -> None:
        """
        保存总览图 / Save a run overview figure

        布局规则：
        1 + number_of_agents 个子图
        Layout rule:
        1 + number_of_agents subplots
        """
        n_panels = 1 + len(agents)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(6 * n_panels, 6),
            squeeze=False,
        )
        axes = axes[0]

        # 子图 0：初始地图 / Panel 0: initial map
        initial_canvas = self._build_canvas(env_map, show_cleaned=False)
        self._draw_background(
            ax=axes[0],
            canvas=initial_canvas,
            env_map=env_map,
            title="Initial Map",
        )
        for idx, agent in enumerate(agents):
            color = self._get_agent_color(idx)
            sx, sy = agent.state.trajectory[0]
            axes[0].scatter(
                sx, sy,
                marker="o",
                s=80,
                color=color,
                label=f"Robot {agent.state.robot_id} start"
            )
        axes[0].legend(loc="upper right")

        # 每个机器人一张单独轨迹图 / One trajectory panel per robot
        for idx, agent in enumerate(agents, start=1):
            final_canvas = self._build_canvas(env_map, show_cleaned=True)
            self._draw_background(
                ax=axes[idx],
                canvas=final_canvas,
                env_map=env_map,
                title=f"Robot {agent.state.robot_id} Trajectory",
            )
            self._draw_agent_trajectory(
                ax=axes[idx],
                agent=agent,
                color=self._get_agent_color(idx - 1),
            )
            axes[idx].legend(loc="upper right")

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

    def _build_canvas(self, env_map: GridMap, show_cleaned: bool) -> np.ndarray:
        """
        构造用于绘图的 RGB 画布 / Build an RGB canvas for plotting
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