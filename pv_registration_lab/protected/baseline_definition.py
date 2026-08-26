"""Protected names and comparison rules for the immutable baseline."""

BASELINE_MODE = "baseline"
IDENTITY_MODE = "identity"


def paired_key(request: dict) -> tuple[str, int, str]:
    return (
        str(request["held_out_station"]),
        int(request["seed"]),
        str(request["stage"]),
    )
