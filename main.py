from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from consts import SESSION_LENGTHS, VALID_SESSIONS
from models import MouseResult, SessionResult, PathStore
from utils import animal_id_and_session_type_from_filename, string_to_ms


def process_session(csv_path: Path) -> SessionResult:
    df = pd.read_csv(csv_path)
    df.columns = map(str.lower, df.columns)

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

    animal_id, session_type = animal_id_and_session_type_from_filename(csv_path)
    return SessionResult(
        animal_id=animal_id,
        session_type=session_type,
        time_investigating=time_investigating,
        number_of_investigations=number_of_investigations,
    )


def get_novel_and_familiar_positions(session_type: str) -> tuple[str, str]:
    match session_type:
        case "Habituation" | "Habituation 1" | "Habituation 2" | "Habituation 3":
            novel = "c"
            familiar = "b"
        case "Sample":
            novel = "a"
            familiar = "b"
        case "Choice":
            novel = "c"
            familiar = "b"
        case _:
            raise ValueError(f"Unknown session type: {session_type}")

    return novel, familiar


def compute_discrimination_index(result: SessionResult) -> float:
    novel, familiar = get_novel_and_familiar_positions(result.session_type)
    return (result.time_investigating[novel] - result.time_investigating[familiar]) / (
        result.time_investigating[novel] + result.time_investigating[familiar]
    )


def plot_discrimination_index(
    mouse_results: list[MouseResult],
) -> None:

    discimination_indexes = {"Habituation": [], "Sample": [], "Choice": []}
    for mouse_result in mouse_results:
        discimination_indexes["Habituation"].append(
            np.mean(
                [
                    compute_discrimination_index(mouse_result.habituation_1),
                    compute_discrimination_index(mouse_result.habituation_2),
                    compute_discrimination_index(mouse_result.habituation_3),
                ]
            )
        )
        discimination_indexes["Sample"].append(
            compute_discrimination_index(mouse_result.sample)
        )
        discimination_indexes["Choice"].append(
            compute_discrimination_index(mouse_result.choice)
        )

    sns.barplot(
        data=pd.DataFrame(discimination_indexes),
        color="lightgray",
        errorbar=None,
        # errorbar="sd",
    )
    sns.stripplot(
        data=pd.DataFrame(discimination_indexes),
        color="black",
        jitter=False,
    )
    plt.title("Discrimination Index")
    plt.axhline(0, color="black")


def plot_results_all_positions(
    session_type: Literal["Habituation", "Sample", "Choice"],
    results: list[SessionResult],
) -> None:
    rows = []

    for result in results:
        for pos in ["a", "b", "c"]:
            rows.append(
                {
                    "subject": result.animal_id,  # ID to connect dots
                    "position": pos,
                    "time": result.time_investigating[pos]
                    / SESSION_LENGTHS[session_type]
                    * 100,
                }
            )

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


def main() -> None:

    path_stores = build_csv_dict()

    results: list[MouseResult] = []
    for path_store in path_stores:
        if path_store.animal_id in ["N001", "N002", "N003"]:
            print(
                f"Skipping {path_store.animal_id} as they are the confounded pilot mice"
            )
            continue
        results.append(
            MouseResult(
                animal_id=path_store.animal_id,
                habituation_1=process_session(path_store.habituation_1),
                habituation_2=process_session(path_store.habituation_2),
                habituation_3=process_session(path_store.habituation_3),
                sample=process_session(path_store.sample),
                choice=process_session(path_store.choice),
            )
        )

    plot_discrimination_index(results)
    # plot_results_all_positions(
    #     session_type="Habituation",
    #     results=[
    #         getattr(result, f"habituation_{i}") for result in results for i in [1, 2, 3]
    #     ],
    # )
    # plot_results_all_positions(
    #     session_type="Sample", results=[result.sample for result in results]
    # )
    # plot_results_all_positions(
    #     session_type="Choice", results=[result.choice for result in results]
    # )

    plt.show()


def parse_list_of_csvs(csvs: list[Path]) -> list[PathStore]:
    """Check that all mice have the expected sessions"""
    store = {}
    for csv in csvs:
        animal_id, session_type = animal_id_and_session_type_from_filename(csv)
        # Deal with N012 and N013 later
        if animal_id in ["N001", "N002", "N003", "N012", "N013"]:
            continue
        if animal_id not in store:
            store[animal_id] = PathStore(animal_id=animal_id)

        setattr(store[animal_id], session_type.lower().replace(" ", "_"), csv)

    path_stores = list(store.values())
    for path_store in path_stores:
        for key in [
            "habituation_1",
            "habituation_2",
            "habituation_3",
            "sample",
            "choice",
        ]:
            assert (
                getattr(path_store, key) is not None
            ), f"Missing {key} for {path_store.animal_id}"
            assert getattr(
                path_store, key
            ).exists(), f"File {getattr(path_store, key)} does not exist for {path_store.animal_id}"
            assert animal_id_and_session_type_from_filename(
                getattr(path_store, key)
            ) == (
                path_store.animal_id,
                key.replace("_", " ").title(),
            ), f"File {getattr(path_store, key)} does not match expected animal id and session type for {path_store.animal_id}"

    return path_stores


def build_csv_dict() -> list[PathStore]:

    suraya_parent = Path("/Volumes/MarcBusche/Suraya/Behaviour/novel_object/")
    jim_parent = Path("/Volumes/MarcBusche/James/novel_object")

    final_day_folders = [
        suraya_parent / "2026-02-24",
        jim_parent / "2026-02-20",
        jim_parent / "2026-02-06",
        jim_parent / "2026-01-30",
    ]

    csvs = []
    for folder in final_day_folders:
        csvs.extend(folder.glob("*.csv"))

    return parse_list_of_csvs(csvs)


if __name__ == "__main__":
    main()
