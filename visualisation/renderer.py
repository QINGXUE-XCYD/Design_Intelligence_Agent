from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from agents.robot_agent import RobotAgent
from environment.grid_map import GridMap

Position = Tuple[int, int]


class MapRenderer:
    """
    地图渲染器 / Map renderer.
    """

    def render_run_overview(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        save_path: str | Path,
    ) -> None:
        n_panels = 1 + len(agents)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(6 * n_panels, 6),
            squeeze=False,
        )
        axes = axes[0]

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

    def plot_battery_curve(
        self,
        step_records: List[dict],
        save_path: str | Path,
        title: str = "Battery Level Over Time",
    ) -> None:
        """
        Plot mean/min battery curves when battery is enabled.
        """
        records = [
            r for r in step_records
            if r.get("mean_battery_level") not in ("", None)
        ]
        if not records:
            return

        steps = [r["step"] for r in records]
        mean_battery = [r["mean_battery_level"] for r in records]
        min_battery = [r["min_battery_level"] for r in records]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, mean_battery, linewidth=2, label="Mean battery")
        ax.plot(steps, min_battery, linewidth=2, linestyle="--", label="Minimum battery")
        ax.set_xlabel("Step")
        ax.set_ylabel("Battery Level")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    def render_cleaning_animation(
        self,
        env_map: GridMap,
        agents: List[RobotAgent],
        step_records: List[Dict],
        agent_step_records: List[Dict],
        save_path: str | Path,
        fps: int = 4,
        max_frames: int = 250,
    ) -> None:
        """
        保存清扫过程 GIF 动图。

        The animation shows cleaned cells, robot positions, trajectories, charging
        stations, coverage, and battery percentage when battery is enabled.
        """
        if not step_records or not agent_step_records:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        step_to_agent_records: Dict[int, List[Dict]] = defaultdict(list)
        for record in agent_step_records:
            step_to_agent_records[int(record["step"])].append(record)

        robot_ids = sorted({int(r["robot_id"]) for r in agent_step_records})
        start_positions = {agent.state.robot_id: agent.state.trajectory[0] for agent in agents}

        raw_frames = [int(r["step"]) for r in step_records]
        stride = max(1, int(np.ceil(len(raw_frames) / max(1, max_frames))))
        frames = raw_frames[::stride]
        if raw_frames[-1] not in frames:
            frames.append(raw_frames[-1])

        cumulative_paths: Dict[int, List[Position]] = {rid: [start_positions[rid]] for rid in robot_ids}
        cumulative_cleaned: set[Position] = set(start_positions.values())

        cleaned_by_frame: Dict[int, set[Position]] = {}
        paths_by_frame: Dict[int, Dict[int, List[Position]]] = {}
        pos_by_frame: Dict[int, Dict[int, Position]] = {}
        battery_by_frame: Dict[int, Dict[int, str]] = {}
        coverage_by_frame: Dict[int, float] = {int(r["step"]): float(r["coverage_rate"]) for r in step_records}

        frame_set = set(frames)
        for step in raw_frames:
            current_positions: Dict[int, Position] = {}
            current_batteries: Dict[int, str] = {}

            for rec in step_to_agent_records.get(step, []):
                rid = int(rec["robot_id"])
                pos = (int(rec["x"]), int(rec["y"]))
                current_positions[rid] = pos
                if cumulative_paths[rid][-1] != pos:
                    cumulative_paths[rid].append(pos)
                cumulative_cleaned.add(pos)

                bp = rec.get("battery_percent", "")
                if bp == "" or bp is None:
                    current_batteries[rid] = ""
                else:
                    try:
                        current_batteries[rid] = f"{float(bp):.0%}"
                    except Exception:
                        current_batteries[rid] = ""

            for rid in robot_ids:
                current_positions.setdefault(rid, cumulative_paths[rid][-1])
                current_batteries.setdefault(rid, "")

            if step in frame_set:
                cleaned_by_frame[step] = set(cumulative_cleaned)
                paths_by_frame[step] = {rid: list(cumulative_paths[rid]) for rid in robot_ids}
                pos_by_frame[step] = dict(current_positions)
                battery_by_frame[step] = dict(current_batteries)

        fig, ax = plt.subplots(figsize=(7, 7))
        image = ax.imshow(self._build_canvas_from_cleaned(env_map, cleaned_by_frame[frames[0]]), origin="lower")
        self._style_animation_axis(ax, env_map)

        # Charging stations.
        if env_map.charging_stations:
            xs = [p[0] for p in env_map.charging_stations]
            ys = [p[1] for p in env_map.charging_stations]
            ax.scatter(xs, ys, marker="s", s=130, facecolors="none", edgecolors="green", linewidths=2, label="Charger")

        robot_artists = {}
        trail_artists = {}
        text_artists = {}
        for idx, rid in enumerate(robot_ids):
            color = self._get_agent_color(idx)
            line, = ax.plot([], [], linewidth=2, alpha=0.85, color=color)
            trail_artists[rid] = line
            scatter = ax.scatter([], [], marker="o", s=90, color=color, label=f"Robot {rid}")
            robot_artists[rid] = scatter
            text_artists[rid] = ax.text(0, 0, "", fontsize=8, ha="center", va="bottom")

        title_artist = ax.set_title("")
        ax.legend(loc="upper right")

        def update(frame_index: int):
            step = frames[frame_index]
            image.set_data(self._build_canvas_from_cleaned(env_map, cleaned_by_frame[step]))

            for rid in robot_ids:
                trail = paths_by_frame[step][rid]
                if trail:
                    xs = [p[0] for p in trail]
                    ys = [p[1] for p in trail]
                    trail_artists[rid].set_data(xs, ys)

                pos = pos_by_frame[step][rid]
                robot_artists[rid].set_offsets(np.array([[pos[0], pos[1]]], dtype=float))

                battery_text = battery_by_frame[step].get(rid, "")
                label = f"R{rid}" + (f" {battery_text}" if battery_text else "")
                text_artists[rid].set_position((pos[0], pos[1] + 0.35))
                text_artists[rid].set_text(label)

            coverage = coverage_by_frame.get(step, 0.0)
            title_artist.set_text(f"Cleaning Progress | Step {step} | Coverage {coverage:.1%}")
            return [image, title_artist, *trail_artists.values(), *robot_artists.values(), *text_artists.values()]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=max(1, int(1000 / max(1, fps))),
            blit=False,
            repeat=False,
        )
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(save_path), writer=writer)
        finally:
            plt.close(fig)

    def _build_canvas(self, env_map: GridMap, show_cleaned: bool) -> np.ndarray:
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
                elif pos in env_map.charging_stations:
                    canvas[y, x] = [0.88, 1.00, 0.88]
                else:
                    canvas[y, x] = [1.0, 1.0, 1.0]

        return canvas

    def _build_canvas_from_cleaned(self, env_map: GridMap, cleaned_cells: set[Position]) -> np.ndarray:
        canvas = np.ones((env_map.height, env_map.width, 3), dtype=float)
        for x in range(env_map.width):
            for y in range(env_map.height):
                pos = (x, y)
                if pos in env_map.static_obstacles:
                    canvas[y, x] = [0.0, 0.0, 0.0]
                elif pos in env_map.dynamic_obstacles:
                    canvas[y, x] = [1.0, 0.6, 0.0]
                elif pos in cleaned_cells:
                    canvas[y, x] = [0.80, 0.92, 1.00]
                elif pos in env_map.charging_stations:
                    canvas[y, x] = [0.88, 1.00, 0.88]
                else:
                    canvas[y, x] = [1.0, 1.0, 1.0]
        return canvas

    def _draw_background(self, ax, canvas: np.ndarray, env_map: GridMap, title: str) -> None:
        ax.imshow(canvas, origin="lower")
        ax.set_title(title)
        ax.set_xlim(-0.5, env_map.width - 0.5)
        ax.set_ylim(-0.5, env_map.height - 0.5)
        ax.set_xticks(range(env_map.width))
        ax.set_yticks(range(env_map.height))
        ax.grid(True, which="both", color="lightgray", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")

    def _style_animation_axis(self, ax, env_map: GridMap) -> None:
        ax.set_xlim(-0.5, env_map.width - 0.5)
        ax.set_ylim(-0.5, env_map.height - 0.5)
        ax.set_xticks(range(env_map.width))
        ax.set_yticks(range(env_map.height))
        ax.grid(True, which="both", color="lightgray", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")

    def _draw_agent_trajectory(self, ax, agent: RobotAgent, color: str) -> None:
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
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return color_cycle[idx % len(color_cycle)]
