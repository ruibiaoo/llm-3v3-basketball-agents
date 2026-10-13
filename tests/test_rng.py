
"""Unit tests for reproducible simulation randomness."""

import pytest

from bball3x3.engine.rng import STREAM_NAMES, GameRNG


def test_all_streams_exist():
    rng = GameRNG.from_seed(42)

    for name in STREAM_NAMES:
        assert rng.get_stream(name) is not None


def test_same_seed_reproducible():
    rng1 = GameRNG.from_seed(42)
    rng2 = GameRNG.from_seed(42)

    for name in STREAM_NAMES:
        values1 = [rng1.random(name) for _ in range(10)]
        values2 = [rng2.random(name) for _ in range(10)]

        assert values1 == values2


def test_different_seeds():
    rng1 = GameRNG.from_seed(42)
    rng2 = GameRNG.from_seed(99)

    values1 = [rng1.random("shooting") for _ in range(10)]
    values2 = [rng2.random("shooting") for _ in range(10)]

    assert values1 != values2


def test_stream_independence():
    rng1 = GameRNG.from_seed(42)
    rng2 = GameRNG.from_seed(42)

    rng1.random("passing")
    rng1.random("passing")

    assert rng1.random("shooting") == rng2.random("shooting")


def test_random_range():
    rng = GameRNG.from_seed(42)

    for name in STREAM_NAMES:
        for _ in range(100):
            value = rng.random(name)
            assert 0.0 <= value < 1.0


def test_bernoulli_zero():
    rng = GameRNG.from_seed(42)

    for _ in range(100):
        assert rng.bernoulli("shooting", 0.0) is False


def test_bernoulli_one():
    rng = GameRNG.from_seed(42)

    for _ in range(100):
        assert rng.bernoulli("shooting", 1.0) is True


def test_invalid_probability():
    rng = GameRNG.from_seed(42)

    with pytest.raises(ValueError):
        rng.bernoulli("shooting", -0.1)

    with pytest.raises(ValueError):
        rng.bernoulli("shooting", 1.1)


def test_choice():
    rng = GameRNG.from_seed(42)
    options = ["A", "B", "C"]

    for _ in range(100):
        assert rng.choice("first_possession", options) in options


def test_choice_reproducible():
    rng1 = GameRNG.from_seed(42)
    rng2 = GameRNG.from_seed(42)

    options = ["A", "B", "C"]

    choices1 = [
        rng1.choice("first_possession", options)
        for _ in range(20)
    ]

    choices2 = [
        rng2.choice("first_possession", options)
        for _ in range(20)
    ]

    assert choices1 == choices2


def test_empty_choice():
    rng = GameRNG.from_seed(42)

    with pytest.raises(ValueError):
        rng.choice("passing", [])


def test_invalid_stream():
    rng = GameRNG.from_seed(42)

    with pytest.raises(ValueError):
        rng.get_stream("unknown")


def test_negative_seed():
    with pytest.raises(ValueError):
        GameRNG.from_seed(-1)
