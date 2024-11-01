from dataclasses import dataclass
from typing import Tuple

@dataclass
class ActionContext:
    window_title: str
    active_element: str
    parent_element: str
    screen_region: Tuple[int, int, int, int]  # (left, top, width, height)