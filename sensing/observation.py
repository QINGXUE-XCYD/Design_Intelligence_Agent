from dataclasses import dataclass, field
from typing import List, Tuple

Position = Tuple[int, int]


@dataclass
class Observation:
    """
    一次局部观测结果 / One local observation result.

    charging_cells records charging stations that are actually visible to the
    robot. This lets public chargers be discovered during exploration instead
    of being globally known from the start.
    """
    visible_cells: List[Position] = field(default_factory=list)
    free_cells: List[Position] = field(default_factory=list)
    occupied_cells: List[Position] = field(default_factory=list)
    dynamic_cells: List[Position] = field(default_factory=list)
    charging_cells: List[Position] = field(default_factory=list)
