from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split

from consts import SESSION_LENGTHS, VALID_SESSIONS
from models import MouseResult, SessionResult, PathStore
from utils import animal_id_and_session_type_from_filename, string_to_ms

sns.set_context("notebook", font_scale=1.2)


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

    df_dict = {
        "genotype": [],
        # "Habituation": [],
        "Sample": [],
        "Choice": [],
    }
    for mouse_result in mouse_results:
        # df_dict["Habituation"].append(
        #     np.mean(
        #         [
        #             compute_discrimination_index(mouse_result.habituation_1),
        #             compute_discrimination_index(mouse_result.habituation_2),
        #             compute_discrimination_index(mouse_result.habituation_3),
        #         ]
        #     )
        # )
        df_dict["Sample"].append(compute_discrimination_index(mouse_result.sample))
        df_dict["Choice"].append(compute_discrimination_index(mouse_result.choice))
        df_dict["genotype"].append(get_genotype(mouse_result.animal_id))

    df = pd.DataFrame(df_dict)
    long_df = df.melt(
        id_vars="genotype",
        # value_vars=["Habituation", "Sample", "Choice"],
        value_vars=["Sample", "Choice"],
        var_name="phase",
        value_name="value",
    )

    ax = plt.gca()

    order = ["Sample", "Choice"]
    hue_order = ["WT", "S305N tau degrader", "NLGF"]

    sns.barplot(
        data=long_df,
        x="phase",
        y="value",
        hue="genotype",
        order=order,
        hue_order=hue_order,
        errorbar=None,
        ax=ax,
        zorder=1,
    )

    bar_handles, bar_labels = ax.get_legend_handles_labels()

    sns.stripplot(
        data=long_df,
        x="phase",
        y="value",
        hue="genotype",
        order=order,
        hue_order=hue_order,
        ax=ax,
        dodge=True,
        zorder=10,
        linewidth=1,
        edgecolor="black",
    )

    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.legend(bar_handles, bar_labels, title="genotype")
    plt.axhline(0, color="black")
    plt.xlabel(None)
    plt.ylabel("Discrimination Index")
    plt.ylim(-0.5, 1)
    plt.tight_layout()


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
        # First pilot that didn't work
        if animal_id in ["N001", "N002", "N003"]:
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


def get_model_for_missing_data() -> None:

    path_stores = build_csv_dict()
    y_list = []
    X_list = []

    for path_store in path_stores:
        df = pd.read_csv(getattr(path_store, "sample"))
        # remove any rows containing NaN values
        df = df.dropna()
        y_list.append(
            np.array(
                [
                    df["Investigating object position a"].to_numpy(),
                    df["Investigating object position B"].to_numpy(),
                    # df["Investigating object position C"].to_numpy(),
                ]
            )
        )

        X_list.append(
            np.array(
                [
                    df["Centre position X"].to_numpy(),
                    df["Centre position Y"].to_numpy(),
                ]
            )
        )

    y_one_hot = np.hstack(y_list)
    n_samples = y_one_hot.shape[1]
    y = np.zeros(n_samples, dtype=int)
    # Identify samples with exactly one active class
    has_class = y_one_hot.sum(axis=0) == 1
    # Assign classes 1–3 based on which row is hot
    y[has_class] = np.argmax(y_one_hot[:, has_class], axis=0) + 1

    y = y.T
    X = np.hstack(X_list).T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=10000,
        C=1,
    )
    print("Fitting model...")
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    predictions = model.predict(X_test)
    accuracy = balanced_accuracy_score(y_test, predictions)
    print(f"Balanced accuracy: {accuracy:.2f}")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])


def fix_missing_data() -> None:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    broken_csvs = [
        file
        for file in Path("/Volumes/MarcBusche/James/novel_object/2026-02-19").glob(
            "*.csv"
        )
        if file.stem.startswith("Unnamed")
    ]

    for csv in broken_csvs:
        df = pd.read_csv(csv)

        X_missing = np.array(
            [
                df["Centre position X"].to_numpy(),
                df["Centre position Y"].to_numpy(),
            ]
        ).T

        predictions = model.predict(X_missing)

        # Create new columns for the predicted investigating positions
        df["Investigating object position a"] = (predictions == 1).astype(int)
        df["Investigating object position B"] = (predictions == 2).astype(int)
        df["Investigating object position C"] = (predictions == 3).astype(int)

        # Save the fixed CSV
        fixed_csv_path = csv.parent / f"fixed_{csv.name}"
        df.to_csv(fixed_csv_path, index=False)
        print(f"Saved fixed CSV to {fixed_csv_path}")


def get_genotype(animal_id: str) -> str:
    match animal_id:
        case "N001" | "N002" | "N003" | "N004" | "N005" | "N006" | "N012" | "N013":
            return "WT"
        case "N009" | "N010" | "N011":
            return "NLGF"
        case (
            "01895"
            | "02244"
            | "02245"
            | "02246"
            | "02247"
            | "02248"
            | "02249"
            | "02252"
        ):
            return "S305N tau degrader"
        case _:
            raise ValueError(f"Unknown animal ID: {animal_id}")


def build_csv_dict() -> list[PathStore]:

    suraya_parent = Path("/Volumes/MarcBusche/Suraya/Behaviour/novel_object/")
    jim_parent = Path("/Volumes/MarcBusche/James/novel_object")

    final_day_folders = [
        suraya_parent / "2026-02-24",
        jim_parent / "2026-02-19",
        jim_parent / "2026-02-06",
        jim_parent / "2026-01-30",
    ]

    csvs = []
    for folder in final_day_folders:
        csvs.extend(folder.glob("*.csv"))

    if len(csvs) == 0:
        raise ValueError("No CSV files found in the specified folders.")
    return parse_list_of_csvs(csvs)


if __name__ == "__main__":
    # fix_missing_data()
    main()
