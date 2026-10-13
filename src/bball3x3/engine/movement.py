"""Movement direction and speed labels (WS1 spec section 9).

``prev_pos`` is stored on every :class:`PlayerState` so the WS3 observation
builder can render Code 1 labels such as ``CUTTING_LEFT``, ``TOWARDS_A1``
or ``FAST``. This module provides the shared derivation helpers.
"""

from bball3x3.config import SimulationConfig

DEFAULT_MAX_CELLS_PER_S = {"SLOW": 1.0, "MEDIUM": 2.0, "FAST": 3.0}


def max_cells_per_s(action_speed: str, config: SimulationConfig) -> float:
    table = {**DEFAULT_MAX_CELLS_PER_S, **config.movement.get("max_cells_per_s", {})}
    return float(table.get(action_speed.upper(), table["FAST"]))


def speed_label(
    prev_pos: tuple[int, int],
    new_pos: tuple[int, int],
    elapsed_s: float,
    movement: str = "HOLD",
) -> str:
    """Derive NONE / SLOW / MEDIUM / FAST from displacement over time."""
    dx = new_pos[0] - prev_pos[0]
    dy = new_pos[1] - prev_pos[1]
    dist_cells = (dx * dx + dy * dy) ** 0.5
    if dist_cells == 0:
        return "NONE" if movement.upper() != "HOLD" else "STATIONARY"
    if elapsed_s <= 0:
        return "SLOW"
    cells_per_s = dist_cells / elapsed_s
    if cells_per_s <= 1.0:
        return "SLOW"
    if cells_per_s <= 2.0:
        return "MEDIUM"
    return "FAST"


def movement_label(
    prev_pos: tuple[int, int],
    new_pos: tuple[int, int],
    prev_basket_dist: float | None = None,
    new_basket_dist: float | None = None,
) -> str:
    """Derive a coarse direction label from a displacement vector."""
    dx = new_pos[0] - prev_pos[0]
    dy = new_pos[1] - prev_pos[1]
    if dx == 0 and dy == 0:
        return "HOLD"
    # Toward the basket (basket sits at low y) reads as a drive/cut.
    if prev_basket_dist is not None and new_basket_dist is not None:
        if new_basket_dist < prev_basket_dist - 0.5:
            if abs(dx) >= abs(dy):
                return "DRIVE_RIGHT" if dx > 0 else "DRIVE_LEFT"
            return "CUTTING_TOWARD_BASKET"
    if abs(dx) >= abs(dy):
        return "CUTTING_RIGHT" if dx > 0 else "CUTTING_LEFT"
    return "CUTTING_UP" if dy > 0 else "CUTTING_DOWN"


def displacement_cells(
    prev_pos: tuple[int, int], new_pos: tuple[int, int]
) -> float:
    dx = new_pos[0] - prev_pos[0]
    dy = new_pos[1] - prev_pos[1]
    return (dx * dx + dy * dy) ** 0.5
