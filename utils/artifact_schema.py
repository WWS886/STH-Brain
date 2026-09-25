"""Read descriptive names from hash-bound external artifact schemas."""
import re


def artifact_name(value):
    """Ignore a trailing revision/date label; preserve the semantic name.

    Callers still verify artifact hashes, rosters, shapes and all provenance.
    This helper never migrates bytes or authenticates an artifact by itself.
    """
    if not isinstance(value, str):
        return value
    return re.sub(r"_(?:v[0-9]+|[0-9]{8})$", "", value)


def schema_matches(value, expected):
    return isinstance(value, str) and artifact_name(value) == expected
