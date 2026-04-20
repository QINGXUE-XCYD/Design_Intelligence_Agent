from enum import Enum


class Action(Enum):
    """
    机器人离散动作 / Discrete robot actions
    """
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4