from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionResult:
    animal_id: str
    session_type: str
    time_investigating: dict[str, float]
    number_of_investigations: dict[str, int]


@dataclass
class MouseResult:
    animal_id: str
    habituation_1: SessionResult
    habituation_2: SessionResult
    habituation_3: SessionResult
    sample: SessionResult
    choice: SessionResult


@dataclass
class PathStore:
    animal_id: str
    habituation_1: Path | None = None
    habituation_2: Path | None = None
    habituation_3: Path | None = None
    sample: Path | None = None
    choice: Path | None = None
