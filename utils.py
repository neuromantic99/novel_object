from pathlib import Path

from consts import VALID_SESSIONS


def animal_id_and_session_type_from_filename(path: Path) -> tuple[str, str]:

    # Bit tricky to get the session type and the mouse id, need to parse the file name
    # think I've caught any potential errors with the assertions but keep and eye on it
    info = path.stem.split(" ")
    idx_session_type = next(
        i for i, s in enumerate(info) if s in ["Habituation", "Sample", "Choice"]
    )
    animal_id = info[idx_session_type - 1]
    assert (
        animal_id[0] == "0" or animal_id[0] == "N"
    ), f"Animal ID should start with 0 for surayas or N for Jimmys, got {animal_id}"
    session_type = " ".join(info[idx_session_type:])
    assert session_type in VALID_SESSIONS, f"Unknown session type: {session_type}"
    return animal_id, session_type


def string_to_ms(time_str: str) -> int:
    """converts a string like this '0:00:00.172' to milliseconds as int"""
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
