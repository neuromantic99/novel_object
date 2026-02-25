from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


VALID_SESSIONS = {"Habituation 1", "Habituation 2", "Habituation 3", "Sample", "Choice"}


@dataclass
class SessionResult:
    animal_id: str
    session_type: str
    time_investigating: dict[str, float]
    number_of_investigations: dict[str, int]


def string_to_ms(time_str: str) -> int:
    """converts a string like this '0:00:00.172' to milliseconds as int"""
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def process_session(csv_path: Path) -> SessionResult:
    df = pd.read_csv(csv_path)
    df.columns = map(str.lower, df.columns)

    # Bit tricky to get the session type and the mouse id, need to parse the file name
    # think I've caught any potential errors with the assertions but keep and eye on it
    info = csv_path.stem.split(" ")
    idx_session_type = next(
        i
        for i, s in enumerate(info)
        if s.lower() in ["habituation", "sample", "choice"]
    )
    animal_id = info[idx_session_type - 1]
    assert (
        animal_id[0] == "0" or animal_id[0] == "N"
    ), f"Animal ID should start with 0 for surayas or N for Jimmys, got {animal_id}"
    session_type = " ".join(info[idx_session_type:])
    assert session_type in VALID_SESSIONS, f"Unknown session type: {session_type}"

    df["time_int"] = df.time.apply(string_to_ms)
    df["dt"] = df.time_int.diff().fillna(0)

    time_investigating = {}
    number_of_investigations = {}

    for position in ["a", "b", "c"]:

        time_investigating[position] = (
            df.loc[df[f"investigating object position {position}"] == 1, "dt"].sum()
            / 1000
        )
        # Count the number of times the mouse started investigating this position
        number_of_investigations[position] = (
            df[f"investigating object position {position}"].diff() == 1
        ).sum()

    return SessionResult(
        animal_id=animal_id,
        session_type=session_type,
        time_investigating=time_investigating,
        number_of_investigations=number_of_investigations,
    )


def plot_results(
    session_type: Literal["Habituation", "Sample", "Choice"],
    results: list[SessionResult],
) -> None:
    rows = []

    divisor = {"Habituation": 20 * 60, "Sample": 8 * 60, "Choice": 3 * 60}

    for result in results:
        for pos in ["a", "b", "c"]:
            rows.append(
                {
                    "subject": result.animal_id,  # ID to connect dots
                    "position": pos,
                    "time": result.time_investigating[pos]
                    / divisor[session_type]
                    * 100,
                }
            )

    print(session_type)
    print(divisor[session_type])

    df = pd.DataFrame(rows)
    # Average across all the same positions
    df = df.groupby(["subject", "position"], as_index=False).mean()
    fig, ax = plt.subplots()

    sns.barplot(
        data=df, x="position", y="time", color="lightgray", ax=ax, errorbar=None
    )

    sns.stripplot(data=df, x="position", y="time", color="black", ax=ax, jitter=False)

    sns.lineplot(
        data=df,
        x="position",
        y="time",
        units="subject",
        estimator=None,
        color="black",
        alpha=0.4,
        ax=ax,
    )

    ax.set_title(session_type)
    ax.set_ylabel("Time Investigating (% total)")
    ax.set_xlabel("Object position")
    plt.ylim(0, 7)


def main(csv_path: Path) -> None:

    all_sessions = [process_session(p) for p in csv_path.glob("*.csv")]

    for session_plot in ("Habituation", "Sample", "Choice"):
        plot_results(
            session_plot,
            [s for s in all_sessions if s.session_type.startswith(session_plot)],
        )
    plt.show()


if __name__ == "__main__":
    path = Path("/Volumes/MarcBusche/Suraya/Behaviour/novel_object/2026-02-24")
    main(path)
