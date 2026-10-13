
"""Court geometry and zone utilities for the 3x3 basketball simulator."""

from math import dist

from bball3x3.config import CourtConfig

ZONES = {
    "TOP": (7, 9),
    "LEFT_WING": (2, 7),
    "RIGHT_WING": (12, 7),
    "LEFT_CORNER": (0, 8),
    "RIGHT_CORNER": (14, 8),
    "LEFT_ELBOW": (5, 5),
    "RIGHT_ELBOW": (9, 5),
    "LEFT_BLOCK": (5, 2),
    "RIGHT_BLOCK": (9, 2),
    "PAINT": (7, 3),
}


class Court:
    """Represent the court using the shared configuration."""

    def __init__(self, config: CourtConfig):
        self.width_m = config.width_m
        self.depth_m = config.depth_m
        self.cell_m = config.cell_m
        self.basket = config.basket
        self.arc_radius_m = config.arc_radius_m

        if self.cell_m <= 0:
            raise ValueError("Cell size must be positive.")

        if self.width_m <= 0 or self.depth_m <= 0:
            raise ValueError("Court dimensions must be positive.")

        self.width_cells = round(self.width_m / self.cell_m)
        self.depth_cells = round(self.depth_m / self.cell_m)

    def is_in_bounds(self, pos: tuple[int, int]) -> bool:
        """Check whether a grid coordinate is inside the court."""
        x, y = pos
        return (
            0 <= x < self.width_cells
            and 0 <= y < self.depth_cells
        )

    def distance(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> float:
        """Calculate Euclidean distance in metres."""
        return dist(a, b) * self.cell_m

    def distance_to_basket(self, pos: tuple[int, int]) -> float:
        """Calculate distance from a position to the basket."""
        return self.distance(pos, self.basket)

    def is_beyond_arc(self, pos: tuple[int, int]) -> bool:
        """Check whether a position is beyond the 2-point arc."""
        if not self.is_in_bounds(pos):
            raise ValueError(f"Position outside court: {pos}")

        return self.distance_to_basket(pos) >= self.arc_radius_m

    def shot_value(self, pos: tuple[int, int]) -> int:
        """Return the point value of a successful field goal."""
        return 2 if self.is_beyond_arc(pos) else 1

    def get_zone_position(self, zone: str) -> tuple[int, int]:
        """Return the coordinates of a named zone."""
        zone = zone.upper()

        if zone not in ZONES:
            raise ValueError(f"Unknown court zone: {zone}")

        position = ZONES[zone]

        if not self.is_in_bounds(position):
            raise ValueError(
                f"Zone {zone} is outside the configured court."
            )

        return position
