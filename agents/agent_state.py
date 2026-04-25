from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

Position = Tuple[int, int]


class AgentMode(Enum):
    """
    机器人当前状态 / Current mode of a robot agent.
    """
    IDLE = "idle"
    EXPLORING = "exploring"
    MOVING = "moving"
    CLEANING = "cleaning"
    RETURNING_TO_CHARGE = "returning_to_charge"
    CHARGING = "charging"
    WAITING_FOR_CHARGE = "waiting_for_charge"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass
class AgentState:
    """
    机器人运行状态容器 / Runtime state container for a robot.
    """
    robot_id: int
    position: Position
    mode: AgentMode = AgentMode.IDLE
    current_goal: Optional[Position] = None
    current_path: List[Position] = field(default_factory=list)
    target_type: Optional[str] = None

    steps_taken: int = 0
    cleaned_cells: int = 0
    idle_steps: int = 0
    total_path_length: float = 0.0

    trajectory: List[Position] = field(default_factory=list)
    done_reason: Optional[str] = None

    # Battery / charging state.
    battery_level: Optional[float] = None
    charging_station: Optional[Position] = None
    assigned_charging_station: Optional[Position] = None
    discovered_charging_stations: Set[Position] = field(default_factory=set)

    # Battery-aware metrics.
    charging_steps: int = 0
    charging_events: int = 0
    charge_wait_steps: int = 0
    low_battery_returns: int = 0
    battery_budget_returns: int = 0
    battery_depletion_count: int = 0
    energy_used: float = 0.0
