"""Reusable UI helpers for manual puzzle-move controls."""


class MoveControlProxy:
    """Proxy used when one compact control represents many internal moves."""

    def __init__(self, widgets):
        self.widgets = widgets

    def configure(self, **kwargs):
        for widget in self.widgets:
            widget.configure(**kwargs)


def square1_manual_move(frame):
    """Build the Square-1 move selected by the manual-control variables."""
    return frame.cube.normalize_move_key(
        (int(frame.square1_u_var.get()), int(frame.square1_d_var.get()), "/" if frame.square1_slash_var.get() else None)
    )


def update_square1_manual_status(frame):
    """Refresh the Square-1 manual-control status label."""
    try:
        move = square1_manual_move(frame)
        status = frame.cube.format_move(move)
        if not frame.cube.is_legal_move(move):
            status += '  illegal'
    except Exception:
        status = 'invalid'
    frame.square1_status_label.configure(text = status)
