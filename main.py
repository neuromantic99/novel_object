from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VALID_SESSIONS = {"Habituation 1", "Habituation 2", "Habituation 3", "Sample", "Choice"}


@dataclass
class SessionResult:
    animal_id: str
    session_type: str
    time_investigating: dict[str, float]


def string_to_ms(time_str: str) -> int:
    """converts a string like this '0:00:00.172' to milliseconds as int"""
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def process_session(csv_path: Path) -> SessionResult:
    df = pd.read_csv(csv_path)
    df.columns = map(str.lower, df.columns)

    # Bit tricky to get the session type and the animal ID due to variable spaces
    # this depends on the animal being called "N0XX"
    info = csv_path.stem.split(" ")
    id_position = next(i for i, s in enumerate(info) if s.startswith("N0"))
    animal_id = info[id_position]
    session_type = " ".join(info[id_position + 1 :])
    assert session_type in VALID_SESSIONS, f"Unknown session type: {session_type}"

    df["time_int"] = df.time.apply(string_to_ms)
    df["dt"] = df.time_int.diff().fillna(0)

    time_investigating = {}

    for position in ["a", "b", "c"]:

        time_investigating[position] = (
            df.loc[df[f"investigating object position {position}"] == 1, "dt"].sum()
            / 1000
        )

    return SessionResult(
        animal_id=animal_id,
        session_type=session_type,
        time_investigating=time_investigating,
    )


def main(csv_path: Path) -> None:

    all_sessions = [process_session(p) for p in csv_path.glob("*.csv")]


if __name__ == "__main__":
    main(Path("/Volumes/MarcBusche/James/novel_object/2026-01-30"))
