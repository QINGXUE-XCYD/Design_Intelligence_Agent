from enum import Enum


class CellType(Enum):
    """
    真实环境中的格子类型 / Cell types in the ground-truth environment
    """
    FREE = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBSTACLE = 2


class CleanState(Enum):
    """
    清扫状态 / Cleaning state
    """
    DIRTY = 0
    CLEANED = 1