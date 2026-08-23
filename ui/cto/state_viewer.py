"""Corner Turning Octahedron state viewer."""

import numpy as np

from ui.fto.state_viewer import FtoStateViewer


class CtoStateViewer(FtoStateViewer):
    """CTO uses the same octahedral net and FTO color palette."""


State_viewer = CtoStateViewer
