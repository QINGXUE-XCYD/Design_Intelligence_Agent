from typing import List, Tuple

from control.action import Action
from environment.grid_map import GridMap, Position


class ActionExecutor:
    """
    动作执行器 / Action executor

    负责将高层路径转换为一步动作，并更新机器人位置。
    Responsible for converting a path into one-step action and updating robot position.
    """

    def next_action_from_path(self, path: List[Position], current_pos: Position) -> Action:
        """
        根据当前路径生成下一步动作 / Generate the next action from current path
        """
        if len(path) < 2:
            return Action.STAY

        next_pos = path[1]
        cx, cy = current_pos
        nx, ny = next_pos

        if nx == cx - 1 and ny == cy:
            return Action.LEFT
        if nx == cx + 1 and ny == cy:
            return Action.RIGHT
        if nx == cx and ny == cy - 1:
            return Action.UP
        if nx == cx and ny == cy + 1:
            return Action.DOWN

        return Action.STAY

    def apply_action(self, current_pos: Position, action: Action) -> Position:
        """
        根据动作计算下一位置 / Compute next position given an action
        """
        x, y = current_pos

        if action == Action.UP:
            return (x, y - 1)
        if action == Action.DOWN:
            return (x, y + 1)
        if action == Action.LEFT:
            return (x - 1, y)
        if action == Action.RIGHT:
            return (x + 1, y)
        return current_pos

    def execute_move(self, current_pos: Position, action: Action, env_map: GridMap) -> Position:
        """
        执行移动 / Execute a movement action

        若目标位置不可通行，则保持原地。
        If target cell is not walkable, stay in place.
        """
        next_pos = self.apply_action(current_pos, action)
        if env_map.is_walkable(next_pos):
            return next_pos
        return current_pos