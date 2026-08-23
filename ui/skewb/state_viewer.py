"""Skewb-specific state viewer."""

import tkinter as Tk


FACE_AXIS = {
    "U": (1, 1),
    "D": (1, -1),
    "R": (0, 1),
    "L": (0, -1),
    "F": (2, 1),
    "B": (2, -1),
}
FACE_CORNER_ORDER = {
    "U": ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)),
    "R": ((1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1)),
    "F": ((-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1)),
    "D": ((-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)),
    "L": ((-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1)),
    "B": ((1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)),
}


class SkewbStateViewer(Tk.Canvas):
    def __init__(self, master, mini_mode = False):
        self.scale = 0.6 if mini_mode else 1.0
        self.face_size = int((54 if mini_mode else 82) * self.scale)
        self.gap = int((5 if mini_mode else 8) * self.scale)
        self.margin = int((14 if mini_mode else 22) * self.scale)
        self.net = {
            "U": (1, 0),
            "L": (0, 1),
            "F": (1, 1),
            "R": (2, 1),
            "B": (3, 1),
            "D": (1, 2),
        }
        width = self.margin * 2 + 4 * self.face_size + 3 * self.gap
        height = self.margin * 2 + 3 * self.face_size + 2 * self.gap
        self.color = {
            "U":"#F5F5F5",
            "R":"#C62828",
            "F":"#2E7D32",
            "D":"#F9D648",
            "L":"#F57C00",
            "B":"#1565C0",
            "":"#7F7F7F",
        }
        Tk.Canvas.__init__(self, master, relief = Tk.RAISED, bd = 2 if mini_mode else 4, width = width, height = height, bg = "#BFBFBF")
        self._draw_face_labels()

    def _face_origin(self, face):
        col, row = self.net[face]
        return (
            self.margin + col * (self.face_size + self.gap),
            self.margin + row * (self.face_size + self.gap),
        )

    def _draw_face_labels(self):
        font = ("Century Gothic", max(8, int(12 * self.scale)), "bold")
        for face in self.net:
            x, y = self._face_origin(face)
            self.create_text(
                x + self.face_size / 2,
                y + self.face_size / 2,
                text = face,
                font = font,
                fill = "#202020",
                tags = "labels",
            )

    def set_color(self, S):
        self.delete("stickers")
        index = 0
        for face in ("U", "R", "F", "D", "L", "B"):
            face_colors = self._oriented_face_colors(face, S[index:index + 5])
            self._draw_face(face, face_colors)
            index += 5
        self.tag_raise("labels")

    def _oriented_face_colors(self, face, raw_colors):
        colors_by_signs = {
            signs:raw_colors[index + 1]
            for index, signs in enumerate(self._face_corner_signs(face))
        }
        return (raw_colors[0],) + tuple(colors_by_signs[signs] for signs in FACE_CORNER_ORDER[face])

    def _face_corner_signs(self, face):
        axis, sign = FACE_AXIS[face]
        signs = []
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    candidate = (x, y, z)
                    if candidate[axis] == sign:
                        signs.append(candidate)
        return tuple(signs)

    def _draw_face(self, face, colors):
        x, y = self._face_origin(face)
        size = self.face_size
        mid = size / 2
        pad = max(1, int(2 * self.scale))
        center_radius = size * 0.23
        polygons = [
            (
                x + pad,
                y + pad,
                x + mid,
                y + mid - center_radius,
                x + mid - center_radius,
                y + mid,
            ),
            (
                x + size - pad,
                y + pad,
                x + mid + center_radius,
                y + mid,
                x + mid,
                y + mid - center_radius,
            ),
            (
                x + size - pad,
                y + size - pad,
                x + mid,
                y + mid + center_radius,
                x + mid + center_radius,
                y + mid,
            ),
            (
                x + pad,
                y + size - pad,
                x + mid - center_radius,
                y + mid,
                x + mid,
                y + mid + center_radius,
            ),
        ]
        for color_key, points in zip(colors[1:], polygons):
            self.create_polygon(points, fill = self.color.get(color_key, "#7F7F7F"), outline = "#202020", tags = "stickers")
        center_points = (
            x + mid,
            y + mid - center_radius,
            x + mid + center_radius,
            y + mid,
            x + mid,
            y + mid + center_radius,
            x + mid - center_radius,
            y + mid,
        )
        self.create_polygon(
            center_points,
            fill = self.color.get(colors[0], "#7F7F7F"),
            outline = "#202020",
            tags = "stickers",
        )


State_viewer = SkewbStateViewer
