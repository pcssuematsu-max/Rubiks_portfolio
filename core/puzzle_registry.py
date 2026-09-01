"""Registry for puzzle-specific model, viewer, notation, and analysis adapters.

The registry deliberately stores factories instead of eagerly imported classes.
This keeps the core module independent from Tkinter and avoids import cycles while
allowing every UI entry point to construct the same puzzle implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple

from core.myperm_effects import MypermEffectAnalyzer


CubeFactory = Callable[[Any], Any]
ViewerFactory = Callable[[Any, Any, bool], Any]
MoveFormatter = Callable[[Any, Iterable[str]], Tuple[str, ...]]
EffectAnalyzer = Callable[[Any, Iterable[str]], Any]


def _default_format_moves(cube, moves):
    """Format a move sequence through the puzzle's common display API."""
    if hasattr(cube, "format_moves"):
        return tuple(cube.format_moves(moves))
    return tuple(moves)


def _default_analyze_effect(cube, moves):
    """Analyze a move sequence with the shared myperm effect analyzer."""
    return MypermEffectAnalyzer(cube).analyze(tuple(moves))


@dataclass(frozen = True)
class PuzzleAdapter:
    """One puzzle's construction, presentation, notation, and analysis hooks."""

    key: str
    title: str
    cube_factory: CubeFactory
    viewer_factory: ViewerFactory
    default_priority_groups: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    format_moves: MoveFormatter = _default_format_moves
    analyze_effect: EffectAnalyzer = _default_analyze_effect

    def create_cube(self, config):
        """Build a new puzzle model from a FrameConfig-compatible object."""
        return self.cube_factory(config)

    def create_viewer(self, master, cube, mini_mode = False):
        """Build this puzzle's normal or compact state viewer."""
        return self.viewer_factory(master, cube, mini_mode)


class PuzzleRegistry:
    """Resolve stable puzzle identifiers to their adapter registrations."""

    def __init__(self):
        self._by_key = {}

    def register(self, adapter):
        """Register an adapter and all of its aliases exactly once."""
        for key in (adapter.key,) + tuple(adapter.aliases):
            normalized_key = self._normalize_key(key)
            if normalized_key in self._by_key:
                raise ValueError(f"puzzle adapter already registered: {key!r}")
            self._by_key[normalized_key] = adapter

    def get(self, key):
        """Return an adapter for *key*, or None for legacy/unregistered puzzles."""
        return self._by_key.get(self._normalize_key(key))

    def adapters(self):
        """Return every adapter once, in registration order."""
        return tuple(dict.fromkeys(self._by_key.values()))

    @staticmethod
    def _normalize_key(key):
        return str(key).strip().lower()


def _create_fto_cube(config):
    from fto.cube import FtoCube

    return FtoCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_cto_cube(config):
    from cto.cube import CtoCube

    return CtoCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_pyraminx_cube(config):
    from pyraminx.cube import PyraminxCube

    return PyraminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_master_pyraminx_cube(config):
    from pyraminx.cube import MasterPyraminxCube

    return MasterPyraminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_skewb_cube(config):
    from skewb.cube import SkewbCube

    return SkewbCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_megaminx_cube(config):
    from megaminx.cube import MegaminxCube

    return MegaminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_square1_cube(config):
    from square1.cube import Square1Cube

    return Square1Cube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_rubiks_cube(config):
    from cube.rubiks_cube import Rubiks_3

    return Rubiks_3(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_fto_viewer(master, cube, mini_mode):
    from ui.fto.state_viewer import FtoStateViewer

    return FtoStateViewer(master, mini_mode = mini_mode)


def _create_cto_viewer(master, cube, mini_mode):
    from ui.cto.state_viewer import CtoStateViewer

    return CtoStateViewer(master, mini_mode = mini_mode)


def _create_pyraminx_viewer(master, cube, mini_mode):
    from ui.pyraminx.state_viewer import PyraminxStateViewer

    return PyraminxStateViewer(master, cube.size, mini_mode = mini_mode)


def _create_skewb_viewer(master, cube, mini_mode):
    from ui.skewb.state_viewer import SkewbStateViewer

    return SkewbStateViewer(master, mini_mode = mini_mode)


def _create_megaminx_viewer(master, cube, mini_mode):
    from ui.megaminx.state_viewer import MegaminxStateViewer

    return MegaminxStateViewer(master, mini_mode = mini_mode)


def _create_square1_viewer(master, cube, mini_mode):
    from ui.square1.state_viewer import Square1StateViewer

    return Square1StateViewer(master, mini_mode = mini_mode)


def _create_rubiks_viewer(master, cube, mini_mode):
    from ui.viewers import StateViewer

    return StateViewer(master, cube.size, mini_mode = mini_mode)


PUZZLE_REGISTRY = PuzzleRegistry()
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "cube",
        aliases = ("rubiks", "rubiks_cube"),
        title = "Rubiks",
        cube_factory = _create_rubiks_cube,
        viewer_factory = _create_rubiks_viewer,
        default_priority_groups = (
            "CoreCenter", "ObliqueCenter-A", "PlusCenter-Layer2",
            "XCenter-Layer2", "ObliqueCenter-B", "PlusCenter-Layer3",
            "XCenter-Layer3", "Wing-Layer2", "Wing-Layer3",
            "Corner", "MidEdge",
        ),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "fto",
        aliases = ("face_turning_octahedron",),
        title = "Face Turning Octahedron",
        cube_factory = _create_fto_cube,
        viewer_factory = _create_fto_viewer,
        default_priority_groups = ("Corner", "Edge", "CenterA", "CenterB"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "megaminx",
        title = "Megaminx",
        cube_factory = _create_megaminx_cube,
        viewer_factory = _create_megaminx_viewer,
        default_priority_groups = ("Corner", "MidEdge"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "square1",
        aliases = ("square-1",),
        title = "Square-1",
        cube_factory = _create_square1_cube,
        viewer_factory = _create_square1_viewer,
        default_priority_groups = ("Corner", "Edge", "Shape"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "pyraminx",
        title = "Pyraminx",
        cube_factory = _create_pyraminx_cube,
        viewer_factory = _create_pyraminx_viewer,
        default_priority_groups = ("Corner", "Edge", "Center"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "master_pyraminx",
        aliases = ("master-pyraminx",),
        title = "Master Pyraminx",
        cube_factory = _create_master_pyraminx_cube,
        viewer_factory = _create_pyraminx_viewer,
        default_priority_groups = ("Corner", "Edge", "MidEdge", "Center"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "skewb",
        title = "Skewb",
        cube_factory = _create_skewb_cube,
        viewer_factory = _create_skewb_viewer,
        default_priority_groups = ("Corner", "Center"),
    )
)
PUZZLE_REGISTRY.register(
    PuzzleAdapter(
        key = "cto",
        aliases = ("corner_turning_octahedron",),
        title = "Corner Turning Octahedron",
        cube_factory = _create_cto_cube,
        viewer_factory = _create_cto_viewer,
        default_priority_groups = ("Corner", "Edge", "Center"),
    )
)


def get_puzzle_adapter(key):
    """Resolve a registered puzzle adapter by key or alias."""
    return PUZZLE_REGISTRY.get(key)
