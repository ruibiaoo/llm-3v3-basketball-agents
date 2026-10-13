
"""Independent, reproducible random streams for the simulation engine."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, SeedSequence

STREAM_NAMES = (
    "shooting",
    "passing",
    "driving",
    "rebounding",
    "contact",
    "free_throws",
    "first_possession",
)


@dataclass
class GameRNG:
    """Manage independent random streams for one simulated game."""

    shooting: Generator
    passing: Generator
    driving: Generator
    rebounding: Generator
    contact: Generator
    free_throws: Generator
    first_possession: Generator

    @classmethod
    def from_seed(cls, seed: int) -> "GameRNG":
        """Create independent RNG streams from a single game seed."""
        if seed < 0:
            raise ValueError("Seed must be non-negative.")

        seed_sequence = SeedSequence(seed)
        child_sequences = seed_sequence.spawn(len(STREAM_NAMES))

        generators = {
            name: np.random.default_rng(child_seed)
            for name, child_seed in zip(
                STREAM_NAMES, child_sequences, strict=True
            )
        }

        return cls(**generators)

    def get_stream(self, name: str) -> Generator:
        """Retrieve a named random stream."""
        if name not in STREAM_NAMES:
            raise ValueError(f"Unknown RNG stream: {name}")

        return getattr(self, name)

    def random(self, stream: str) -> float:
        """Draw a random float in [0, 1) from a named stream."""
        return float(self.get_stream(stream).random())

    def bernoulli(self, stream: str, probability: float) -> bool:
        """Return True with the specified probability."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")

        return self.random(stream) < probability

    def choice(self, stream: str, options):
        """Randomly select one item from a non-empty sequence."""
        if len(options) == 0:
            raise ValueError("Cannot choose from an empty sequence.")

        index = int(
            self.get_stream(stream).integers(0, len(options))
        )
        return options[index]
