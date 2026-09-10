"""Registry for puzzle-specific model, viewer, notation, and analysis adapters.

The registry deliberately stores factories instead of eagerly imported classes.
This keeps the core module independent from Tkinter and avoids import cycles while
allowing every UI entry point to construct the same puzzle implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, cast

try:
    from typing import Protocol
except ImportError:  # Python 3.7 compatibility
    from typing_extensions import Protocol

from core.myperm_effects import MypermEffect, MypermEffectAnalyzer


Square1Move = Tuple[int, int, Optional[str]]
PuzzleMove = Union[str, Square1Move]
MoveSequence = Iterable[PuzzleMove]
DisplayMoveSequence = Tuple[str, ...]


class PuzzleConfig(Protocol):
    """Configuration fields required by every registered puzzle factory."""

    cube_size: int
    F2L: bool
    OLL: bool
    Centers: bool
    Edges: bool
    Cross: bool


class PuzzleModel(Protocol):
    """Minimum state-model contract shared by the registered puzzles."""

    size: int


CubeFactory = Callable[[PuzzleConfig], PuzzleModel]
ViewerFactory = Callable[[object, PuzzleModel, bool], object]
MoveFormatter = Callable[[PuzzleModel, MoveSequence], DisplayMoveSequence]
EffectAnalyzer = Callable[[PuzzleModel, MoveSequence], MypermEffect]


def _default_format_moves(cube: PuzzleModel, moves: MoveSequence) -> DisplayMoveSequence:
    """Format a move sequence through the puzzle's common display API."""
    formatter = cast(
        Optional[Callable[[MoveSequence], Iterable[str]]],
        getattr(cube, "format_moves", None),
    )
    if formatter is not None:
        return tuple(formatter(moves))
    return tuple(cast(str, move) for move in moves)


def _default_analyze_effect(cube: PuzzleModel, moves: MoveSequence) -> MypermEffect:
    """Analyze a move sequence with the shared myperm effect analyzer."""
    return cast(MypermEffect, MypermEffectAnalyzer(cube).analyze(tuple(moves)))


@dataclass(frozen = True)
class PuzzleAdapter:
    """One puzzle's construction, presentation, notation, and analysis hooks."""

    key: str
    title: str
    cube_factory: CubeFactory
    viewer_factory: ViewerFactory
    default_priority_groups: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    format_moves: MoveFormatter = _default_format_moves
    analyze_effect: EffectAnalyzer = _default_analyze_effect

    def __post_init__(self) -> None:
        """Reject malformed registrations at the registry boundary."""
        if not isinstance(self.key, str) or not self.key.strip():
            raise TypeError("PuzzleAdapter.key must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise TypeError("PuzzleAdapter.title must be a non-empty string")
        if not isinstance(self.aliases, tuple):
            raise TypeError("PuzzleAdapter.aliases must be a tuple of strings")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise TypeError("PuzzleAdapter.aliases must contain non-empty strings")
        if not isinstance(self.default_priority_groups, tuple):
            raise TypeError("PuzzleAdapter.default_priority_groups must be a tuple of strings")
        if any(not isinstance(group, str) or not group.strip() for group in self.default_priority_groups):
            raise TypeError("PuzzleAdapter.default_priority_groups must contain non-empty strings")
        for name, callback in (
            ("cube_factory", self.cube_factory),
            ("viewer_factory", self.viewer_factory),
            ("format_moves", self.format_moves),
            ("analyze_effect", self.analyze_effect),
        ):
            if not callable(callback):
                raise TypeError(f"PuzzleAdapter.{name} must be callable")

    def create_cube(self, config: PuzzleConfig) -> PuzzleModel:
        """Build a new puzzle model from a FrameConfig-compatible object."""
        return self.cube_factory(config)

    def create_viewer(self, master: object, cube: PuzzleModel, mini_mode: bool = False) -> object:
        """Build this puzzle's normal or compact state viewer."""
        return self.viewer_factory(master, cube, mini_mode)


class PuzzleRegistry:
    """Resolve stable puzzle identifiers to their adapter registrations."""

    def __init__(self) -> None:
        self._by_key: Dict[str, PuzzleAdapter] = {}

    def register(self, adapter: PuzzleAdapter) -> None:
        """Register an adapter and all of its aliases exactly once."""
        for key in (adapter.key,) + tuple(adapter.aliases):
            normalized_key = self._normalize_key(key)
            if normalized_key in self._by_key:
                raise ValueError(f"puzzle adapter already registered: {key!r}")
            self._by_key[normalized_key] = adapter

    def get(self, key: object) -> Optional[PuzzleAdapter]:
        """Return an adapter for *key*, or None for legacy/unregistered puzzles."""
        return self._by_key.get(self._normalize_key(key))

    def adapters(self) -> Tuple[PuzzleAdapter, ...]:
        """Return every adapter once, in registration order."""
        return tuple(dict.fromkeys(self._by_key.values()))

    @staticmethod
    def _normalize_key(key: object) -> str:
        return str(key).strip().lower()


def _create_fto_cube(config: PuzzleConfig) -> PuzzleModel:
    from fto.cube import FtoCube

    return FtoCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_cto_cube(config: PuzzleConfig) -> PuzzleModel:
    from cto.cube import CtoCube

    return CtoCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_pyraminx_cube(config: PuzzleConfig) -> PuzzleModel:
    from pyraminx.cube import PyraminxCube

    return PyraminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_master_pyraminx_cube(config: PuzzleConfig) -> PuzzleModel:
    from pyraminx.cube import MasterPyraminxCube

    return MasterPyraminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_skewb_cube(config: PuzzleConfig) -> PuzzleModel:
    from skewb.cube import SkewbCube

    return SkewbCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_megaminx_cube(config: PuzzleConfig) -> PuzzleModel:
    from megaminx.cube import MegaminxCube

    return MegaminxCube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_square1_cube(config: PuzzleConfig) -> PuzzleModel:
    from square1.cube import Square1Cube

    return Square1Cube(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_rubiks_cube(config: PuzzleConfig) -> PuzzleModel:
    from cube.rubiks_cube import Rubiks_3

    return Rubiks_3(
        size = config.cube_size,
        F2L = config.F2L,
        OLL = config.OLL,
        Centers = config.Centers,
        Edges = config.Edges,
        Cross = config.Cross,
    )


def _create_fto_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.fto.state_viewer import FtoStateViewer

    return FtoStateViewer(master, mini_mode = mini_mode)


def _create_cto_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.cto.state_viewer import CtoStateViewer

    return CtoStateViewer(master, mini_mode = mini_mode)


def _create_pyraminx_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.pyraminx.state_viewer import PyraminxStateViewer

    return PyraminxStateViewer(master, cube.size, mini_mode = mini_mode)


def _create_skewb_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.skewb.state_viewer import SkewbStateViewer

    return SkewbStateViewer(master, mini_mode = mini_mode)


def _create_megaminx_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.megaminx.state_viewer import MegaminxStateViewer

    return MegaminxStateViewer(master, mini_mode = mini_mode)


def _create_square1_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
    from ui.square1.state_viewer import Square1StateViewer

    return Square1StateViewer(master, mini_mode = mini_mode)


def _create_rubiks_viewer(master: object, cube: PuzzleModel, mini_mode: bool) -> object:
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


def get_puzzle_adapter(key: object) -> Optional[PuzzleAdapter]:
    """Resolve a registered puzzle adapter by key or alias."""
    return PUZZLE_REGISTRY.get(key)
