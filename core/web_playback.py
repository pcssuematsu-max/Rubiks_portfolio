"""Build links from a GUI solve session to the public 3D playback page."""

from __future__ import annotations

from urllib.parse import urlencode


WEB_PLAYBACK_URL = (
    "https://pcssuematsu-max.github.io/github.io/"
    "twisty-puzzle-ai-lab-playback.html"
)
OWN_3D_VIEWER_URL = (
    "https://pcssuematsu-max.github.io/github.io/"
    "3d-puzzle-viewer.html"
)


def web_puzzle_key(puzzle_type: str, cube_size: int) -> str | None:
    """Return the cubing.js puzzle key for a supported GUI puzzle."""
    normalized_type = str(puzzle_type).strip().lower()
    if normalized_type in {"cube", "rubiks", "rubiks_cube"}:
        size = int(cube_size)
        if 2 <= size <= 7:
            return f"{size}x{size}x{size}"
        return None
    return {
        "megaminx": "megaminx",
        "pyraminx": "pyraminx",
        "skewb": "skewb",
        "square1": "square1",
        "fto": "fto",
    }.get(normalized_type)


def build_web_playback_url(puzzle: str, moves, setup=()) -> str:
    """Create a shareable playback URL from display-notation move sequences."""
    query = {
        "puzzle": puzzle,
        "moves": " ".join(str(move).strip() for move in moves if str(move).strip()),
    }
    setup_text = " ".join(str(move).strip() for move in setup if str(move).strip())
    if setup_text:
        query["setup"] = setup_text
    return WEB_PLAYBACK_URL + "?" + urlencode(query)


def own_viewer_puzzle_key(puzzle_type: str, cube_size: int) -> str | None:
    """Return the self-built viewer key for a supported GUI puzzle.

    Keeping this conversion separate from ``web_puzzle_key`` leaves the
    cubing.js page available for non-cube puzzles it already supports.
    """
    normalized_type = str(puzzle_type).strip().lower()
    size = int(cube_size)
    if normalized_type in {"cube", "rubiks", "rubiks_cube"} and 2 <= size <= 7:
        return f"cube-{size}x{size}"
    return None


def build_own_3d_viewer_url(
    puzzle: str,
    moves,
    setup=(),
    position: int = 0,
    theme: str | None = None,
) -> str:
    """Create a self-built 3D viewer URL from display-notation move sequences."""
    query = {"puzzle": puzzle}
    moves_text = " ".join(str(move).strip() for move in moves if str(move).strip())
    setup_text = " ".join(str(move).strip() for move in setup if str(move).strip())
    if moves_text:
        query["moves"] = moves_text
    if setup_text:
        query["setup"] = setup_text
    if int(position) > 0:
        query["position"] = str(int(position))
    if theme:
        query["theme"] = str(theme)
    return OWN_3D_VIEWER_URL + "?" + urlencode(query)
