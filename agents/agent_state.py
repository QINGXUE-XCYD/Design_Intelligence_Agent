from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from environment.grid_map import Position


class AgentRole(Enum):
    """
    机器人角色 / Robot role
    """
    SCOUT = "scout"
    CLEANER = "cleaner"


class AgentMode(Enum):
    """
    机器人当前状态 / Current mode of a robot agent
    """
    WAITING = "waiting"
    IDLE = "idle"
    EXPLORING = "exploring"
    MOVING = "moving"
    CLEANING = "cleaning"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass
class AgentState:
    """
    机器人运行状态容器 / Runtime state container for a robot
    """
    robot_id: int
    position: Position
    role: AgentRole = AgentRole.CLEANER
    mode: AgentMode = AgentMode.EXPLORING
    current_goal: Optional[Position] = None
    current_path: List[Position] = field(default_factory=list)

    steps_taken: int = 0
    cleaned_cells: int = 0
    idle_steps: int = 0
    total_path_length: float = 0.0

    # 新增：轨迹记录 / Added: trajectory history
    trajectory: List[Position] = field(default_factory=list)
    # 新增：agent 自身结束原因 / Added: per-agent termination reason
    done_reason: Optional[str] = None
    activation_step: Optional[int] = None
