"""Build links from a GUI solve session to the public 3D playback page."""

from __future__ import annotations

from urllib.parse import urlencode


WEB_PLAYBACK_URL = (
    "https://pcssuematsu-max.github.io/github.io/"
    "twisty-puzzle-ai-lab-playback.html"
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
