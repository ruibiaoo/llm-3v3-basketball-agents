
import pytest

from bball3x3.config import CourtConfig
from bball3x3.engine.court import ZONES, Court


@pytest.fixture
def court():
    return Court(CourtConfig())


def test_court_dimensions(court):
    assert court.width_cells == 15
    assert court.depth_cells == 11


def test_court_bounds(court):
    assert court.is_in_bounds((0, 0))
    assert court.is_in_bounds((14, 10))
    assert not court.is_in_bounds((-1, 5))
    assert not court.is_in_bounds((15, 5))
    assert not court.is_in_bounds((7, 11))


def test_distance(court):
    assert court.distance((0, 0), (3, 4)) == 5.0


def test_distance_to_basket(court):
    assert court.distance_to_basket((7, 1)) == 0.0


def test_arc(court):
    assert not court.is_beyond_arc((7, 3))
    assert court.is_beyond_arc((7, 9))


def test_shot_value(court):
    assert court.shot_value((7, 3)) == 1
    assert court.shot_value((7, 9)) == 2


def test_zones_in_bounds(court):
    for position in ZONES.values():
        assert court.is_in_bounds(position)


def test_zone_lookup(court):
    assert court.get_zone_position("TOP") == (7, 9)
    assert court.get_zone_position("top") == (7, 9)


def test_invalid_zone(court):
    with pytest.raises(ValueError):
        court.get_zone_position("UNKNOWN")


def test_invalid_position(court):
    with pytest.raises(ValueError):
        court.is_beyond_arc((20, 20))


def test_invalid_cell_size():
    with pytest.raises(ValueError):
        Court(CourtConfig(cell_m=0))
