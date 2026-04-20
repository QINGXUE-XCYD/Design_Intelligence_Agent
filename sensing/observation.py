from dataclasses import dataclass, field
from typing import List, Tuple

Position = Tuple[int, int]


@dataclass
class Observation:
    """
    一次局部观测结果 / One local observation result

    将感知结果与地图更新解耦。
    Decouples sensing result from map updating.
    """
    visible_cells: List[Position] = field(default_factory=list)
    free_cells: List[Position] = field(default_factory=list)
    occupied_cells: List[Position] = field(default_factory=list)
    dynamic_cells: List[Position] = field(default_factory=list)