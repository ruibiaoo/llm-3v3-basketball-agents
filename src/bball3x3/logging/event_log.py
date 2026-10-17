"""Write and read one game's typed JSONL event records.

This module validates individual records and game_start/game_end boundaries.
It does not check basketball outcomes, event counts or score reconciliation.
"""

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Self, TypeAlias

from pydantic import TypeAdapter

from bball3x3.schemas.log import GameEndRecord, GameStartRecord, LogModel, LogRecord

_RECORD_ADAPTER = TypeAdapter(LogRecord)
_KNOWN_RECORD_TYPES = frozenset(
    {
        "game_start",
        "decision",
        "message",
        "captain_call",
        "memory_proposal",
        "memory_write",
        "cs_proposal",
        "cs_write",
        "resolution",
        "referee",
        "state",
        "invariant_violation",
        "possession_end",
        "game_end",
    }
)
ReadRecord: TypeAlias = LogRecord | dict[str, Any]


class LogReadError(ValueError):
    """An invalid log line, with its path and one-based line number."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


class IncompleteLogError(LogReadError):
    """The file is empty or ended before a normal game_end record."""


def _check_boundaries(record: ReadRecord, started: bool, ended: bool) -> None:
    if ended:
        raise ValueError("No records are allowed after game_end.")
    if not started and not isinstance(record, GameStartRecord):
        raise ValueError("The first record must be game_start.")
    if started and isinstance(record, GameStartRecord):
        raise ValueError("Only one game_start is allowed per file.")


class EventLogger:
    """Create a new log and write records from a single game runner.

    Parent directories are created as needed. Existing files are never
    overwritten or resumed. Each write is flushed to the OS so another reader
    can inspect it; this is not an fsync guarantee against power loss.

    Closing the logger never invents a game_end record. The runner must supply
    that record when the game actually finishes. This class is not thread-safe.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", newline="\n")
        self._started = False
        self._ended = False

    @property
    def closed(self) -> bool:
        return self._file.closed

    def write(self, record: LogRecord) -> None:
        """Validate and append one record; never write part of invalid input.

        Schema/serialization errors and I/O errors propagate to the caller.
        Models are revalidated from plain data because they may have been
        mutated since construction, including their nested models.
        """
        self._ensure_open()
        if not isinstance(record, LogModel):
            raise TypeError("write() requires a typed log record, not a dictionary.")

        validated = _RECORD_ADAPTER.validate_python(
            record.model_dump(mode="python", warnings="error")
        )
        _check_boundaries(validated, self._started, self._ended)
        line = json.dumps(
            # Keep floats unchanged until json.dumps checks them. Pydantic's
            # JSON mode can turn NaN/Infinity inside untyped metadata into null.
            validated.model_dump(mode="python", warnings="error"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._file.write(line + "\n")
        self._started = True
        self._ended = isinstance(validated, GameEndRecord)
        self.flush()

    def flush(self) -> None:
        """Flush buffered records without ending the game."""
        self._ensure_open()
        self._file.flush()

    def close(self) -> None:
        """Flush and close the file; repeated calls are harmless."""
        self._file.close()

    def _ensure_open(self) -> None:
        if self.closed:
            raise ValueError("The event logger is closed.")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed.")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number {value!r} exceeds the finite float range.")
    return number


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def read_events(
    path: str | Path,
    *,
    require_complete: bool = True,
    allow_unknown: bool = False,
) -> Iterator[ReadRecord]:
    """Yield typed records, raising LogReadError for bad input with line context.

    Consume the iterator to EOF to check the complete file. By default a missing
    game_end raises IncompleteLogError. For inspection of a still-running or
    interrupted game, pass require_complete=False. Unknown record types are
    rejected by default; with allow_unknown=True they are preserved as
    dictionaries, after basic record and event-index checks.

    A final valid JSON object without a trailing newline is accepted. Ordinary
    file-access errors propagate unchanged. Whole-game metric validation is a
    separate step, even when a game_end is present.
    """
    log_path = Path(path)
    started = False
    ended = False
    line_number = 0

    # Decode each binary line separately so UTF-8 failures have accurate line
    # numbers rather than being raised while a text buffer reads ahead.
    with log_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                data = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                    object_pairs_hook=_unique_object,
                )
                if (
                    allow_unknown
                    and isinstance(data, dict)
                    and isinstance(data.get("record_type"), str)
                    and data["record_type"] not in _KNOWN_RECORD_TYPES
                ):
                    if not data["record_type"]:
                        raise ValueError("Unknown record_type must be a non-empty string.")
                    event_idx = data.get("event_idx")
                    if event_idx is not None and (
                        isinstance(event_idx, bool)
                        or not isinstance(event_idx, int)
                        or event_idx < 1
                    ):
                        raise ValueError("Unknown record event_idx must be a positive integer.")
                    record: ReadRecord = data
                else:
                    record = _RECORD_ADAPTER.validate_python(data)
                _check_boundaries(record, started, ended)
            except ValueError as exc:
                raise LogReadError(log_path, line_number, str(exc)) from exc

            started = True
            ended = isinstance(record, GameEndRecord)
            yield record

    if not started:
        raise IncompleteLogError(log_path, 1, "Empty log; game_start is missing.")
    if require_complete and not ended:
        raise IncompleteLogError(log_path, line_number + 1, "Missing game_end at EOF.")
