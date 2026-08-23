"""Pyraminx-specific state viewer."""

import tkinter as Tk


class PyraminxStateViewer(Tk.Canvas):
    def __init__(self, master, order = 3, mini_mode = False):
        self.order = int(order)
        self.scale = 0.6 if mini_mode else 1.0
        cell_size = 32 if self.order >= 4 else 38
        if mini_mode:
            cell_size = 24
        self.face_side = cell_size * self.order * self.scale
        self.face_height = self.face_side * 0.8660254038
        self.margin = 28 * self.scale
        self.face_triangles = self._build_face_triangles()
        width, height = self._canvas_size()
        self.color = {
            "U":"#7F0000",
            "L":"#007F00",
            "R":"#0000BF",
            "B":"#BFBF00",
            "":"#7F7F7F",
        }
        Tk.Canvas.__init__(self, master, relief = Tk.RAISED, bd = 2 if mini_mode else 4, width = width, height = height, bg = "#BFBFBF")
        self._draw_face_labels()

    def _build_face_triangles(self):
        corner = {
            "B": (0.0, 0.0),
            "L": (-self.face_side / 2, self.face_height),
            "R": (self.face_side / 2, self.face_height),
        }
        corner["U_L"] = self._reflect_point(corner["L"], corner["R"], corner["B"])
        corner["U_R"] = self._reflect_point(corner["R"], corner["B"], corner["L"])
        corner["U_B"] = self._reflect_point(corner["B"], corner["L"], corner["R"])

        triangles = {
            # Face names are opposite-corner names. Keep the same vertex order
            # as pyraminx.cube.PyraminxCube.face_vertices.
            "U": (corner["L"], corner["B"], corner["R"]),
            "L": (corner["U_L"], corner["R"], corner["B"]),
            "R": (corner["U_R"], corner["B"], corner["L"]),
            "B": (corner["U_B"], corner["L"], corner["R"]),
        }
        return self._shift_triangles_into_canvas(triangles)

    def _reflect_point(self, point, line_a, line_b):
        px, py = point
        ax, ay = line_a
        bx, by = line_b
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        foot = (ax + t * dx, ay + t * dy)
        return (2 * foot[0] - px, 2 * foot[1] - py)

    def _shift_triangles_into_canvas(self, triangles):
        xs = [point[0] for triangle in triangles.values() for point in triangle]
        ys = [point[1] for triangle in triangles.values() for point in triangle]
        shift_x = self.margin - min(xs)
        shift_y = self.margin - min(ys) + 18 * self.scale
        return {
            face: tuple((point[0] + shift_x, point[1] + shift_y) for point in triangle)
            for face, triangle in triangles.items()
        }

    def _canvas_size(self):
        xs = [point[0] for triangle in self.face_triangles.values() for point in triangle]
        ys = [point[1] for triangle in self.face_triangles.values() for point in triangle]
        return int(max(xs) + self.margin), int(max(ys) + self.margin)

    def _draw_face_labels(self):
        font = ("Century Gothic", max(8, int(14 * self.scale)), "bold")
        for face, triangle in self.face_triangles.items():
            x = sum(point[0] for point in triangle) / 3
            y = sum(point[1] for point in triangle) / 3
            self.create_text(x, y, text = face, font = font, fill = "#202020", tags = "labels")

    def set_color(self, S):
        self.delete("stickers")
        index = 0
        for face in ("U", "L", "R", "B"):
            triangle = self.face_triangles[face]
            for i in range(self.order):
                for j in range(self.order - i):
                    self._draw_triangle(triangle, i, j, "up", S[index])
                    index += 1
                    if i + j < self.order - 1:
                        self._draw_triangle(triangle, i, j, "down", S[index])
                        index += 1
        self.tag_raise("labels")

    def _draw_triangle(self, face_triangle, i, j, orientation, color_key):
        a = self._face_point(face_triangle, i, j)
        if orientation == "up":
            b = self._face_point(face_triangle, i + 1, j)
            c = self._face_point(face_triangle, i, j + 1)
        else:
            a = self._face_point(face_triangle, i + 1, j)
            b = self._face_point(face_triangle, i + 1, j + 1)
            c = self._face_point(face_triangle, i, j + 1)
        points = (a[0], a[1], b[0], b[1], c[0], c[1])
        fill = self.color.get(color_key, "#7F7F7F")
        self.create_polygon(points, fill = fill, outline = "#202020", tags = "stickers")

    def _face_point(self, face_triangle, i, j):
        top, left, right = face_triangle
        u = i / self.order
        v = j / self.order
        w = 1.0 - u - v
        return (
            w * top[0] + u * left[0] + v * right[0],
            w * top[1] + u * left[1] + v * right[1],
        )


State_viewer = PyraminxStateViewer
