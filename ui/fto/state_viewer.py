"""Face Turning Octahedron state viewer."""

import math
import tkinter as Tk

import numpy as np

from fto.cube import FTO_FACE_NAMES, FTO_FACE_SIGNS


class FtoStateViewer(Tk.Canvas):
    def __init__(self, master, mini_mode = False):
        self.scale = 0.58 if mini_mode else 1.0
        self.face_size = int(74 if mini_mode else 112)
        self.margin = int((12 if mini_mode else 20) * self.scale)
        self.axis_names = ("x", "y", "z")
        self.face_vertices = self._build_face_vertices()
        self._shift_net_to_margin()
        width, height = self._canvas_size()
        self.color = {
            "A":"#5FBF00",
            "B":"#3F007F",
            "C":"#FF7F00",
            "D":"#0000BF",
            "E":"#7F0000",
            "F":"#005F2F",
            "G":"#DFDF00",
            "H":"#007FFF",
            "":"#7F7F7F",
        }
        Tk.Canvas.__init__(self, master, relief = Tk.RAISED, bd = 2 if mini_mode else 4, width = width, height = height, bg = "#BFBFBF")
        self._draw_face_labels()

    def _build_face_vertices(self):
        vertices = self._build_unmirrored_face_vertices()
        return self._mirror_vertices_horizontally(vertices)

    def _build_unmirrored_face_vertices(self):
        side = self.face_size * self.scale
        height = side * math.sqrt(3.0) / 2.0
        vertices = {
            "URF":{
                "x":np.array([side / 2.0, 0.0]),
                "y":np.array([side, height]),
                "z":np.array([0.0, height]),
            },
        }
        unfold_edges = (
            ("URF", "UFL"),
            ("URF", "UBR"),
            ("URF", "DFR"),
            ("UFL", "ULB"),
            ("UFL", "DLF"),
            ("UBR", "DRB"),
            ("DLF", "DBL"),
        )
        for source_face, target_face in unfold_edges:
            vertices[target_face] = self._reflected_neighbor_vertices(vertices[source_face], source_face, target_face)
        return vertices

    def _mirror_vertices_horizontally(self, vertices):
        all_x = [
            point[0]
            for face_vertices in vertices.values()
            for point in face_vertices.values()
        ]
        mirror_axis = min(all_x) + max(all_x)
        return {
            face:{
                axis:np.array([mirror_axis - point[0], point[1]])
                for axis, point in face_vertices.items()
            }
            for face, face_vertices in vertices.items()
        }

    def _reflected_neighbor_vertices(self, source_vertices, source_face, target_face):
        changed_axis = self._changed_axis(source_face, target_face)
        shared_axes = [axis for axis in self.axis_names if axis != changed_axis]
        neighbor_vertices = {
            shared_axes[0]:source_vertices[shared_axes[0]],
            shared_axes[1]:source_vertices[shared_axes[1]],
        }
        neighbor_vertices[changed_axis] = self._reflect_point(
            source_vertices[changed_axis],
            source_vertices[shared_axes[0]],
            source_vertices[shared_axes[1]],
        )
        return neighbor_vertices

    def _changed_axis(self, source_face, target_face):
        source_signs = FTO_FACE_SIGNS[source_face]
        target_signs = FTO_FACE_SIGNS[target_face]
        for index, axis in enumerate(self.axis_names):
            if source_signs[index] != target_signs[index]:
                return axis
        raise ValueError((source_face, target_face))

    def _reflect_point(self, point, edge_start, edge_end):
        edge_vector = edge_end - edge_start
        point_vector = point - edge_start
        projection = edge_start + edge_vector * np.dot(point_vector, edge_vector) / np.dot(edge_vector, edge_vector)
        return 2.0 * projection - point

    def _shift_net_to_margin(self):
        all_points = np.array([
            point
            for vertices in self.face_vertices.values()
            for point in vertices.values()
        ])
        offset = np.array([self.margin, self.margin]) - all_points.min(axis = 0)
        for face in self.face_vertices:
            for axis in self.face_vertices[face]:
                self.face_vertices[face][axis] = self.face_vertices[face][axis] + offset

    def _canvas_size(self):
        all_points = np.array([
            point
            for vertices in self.face_vertices.values()
            for point in vertices.values()
        ])
        max_point = all_points.max(axis = 0)
        return int(max_point[0] + self.margin), int(max_point[1] + self.margin)

    def _draw_face_labels(self):
        font = ("Century Gothic", max(7, int(10 * self.scale)), "bold")
        for face in FTO_FACE_NAMES:
            cx, cy = self._face_center(face)
            self.create_text(cx, cy, text = face, font = font, fill = "#202020", tags = "labels")

    def _face_center(self, face):
        points = list(self.face_vertices[face].values())
        center = sum(points) / 3.0
        return center[0], center[1]

    def set_color(self, S):
        self.delete("stickers")
        index = 0
        for face in FTO_FACE_NAMES:
            self._draw_face(face, S[index:index + 9])
            index += 9

    def _draw_face(self, face, colors):
        vertices = self._display_vertices(face)
        triangles = self._subtriangle_barycentrics(3)
        for color_key, triangle in zip(colors, triangles):
            points = []
            for barycentric in triangle:
                px, py = self._barycentric_point(vertices, barycentric)
                points.extend([px, py])
            self.create_polygon(
                points,
                fill = self.color.get(color_key, "#7F7F7F"),
                outline = "#202020",
                width = max(1, int(1 * self.scale)),
                tags = "stickers",
            )
        self._draw_face_outline(vertices)

    def _display_vertices(self, face):
        vertices = self.face_vertices[face]
        return tuple((vertices[axis][0], vertices[axis][1]) for axis in self.axis_names)

    def _draw_face_outline(self, vertices):
        points = []
        for point in vertices:
            points.extend(point)
        self.create_polygon(
            points,
            fill = "",
            outline = "#111111",
            width = max(1, int(2 * self.scale)),
            tags = "stickers",
        )

    def _subtriangle_barycentrics(self, order):
        triangles = []
        for i in range(order):
            for j in range(order - i):
                k = order - 1 - i - j
                triangles.append(((i + 1, j, k), (i, j + 1, k), (i, j, k + 1)))
        for i in range(order - 1):
            for j in range(order - 1 - i):
                k = order - 2 - i - j
                triangles.append(((i + 1, j + 1, k), (i + 1, j, k + 1), (i, j + 1, k + 1)))
        return triangles

    def _barycentric_point(self, vertices, barycentric):
        denominator = float(sum(barycentric))
        x = sum(barycentric[index] * vertices[index][0] for index in range(3)) / denominator
        y = sum(barycentric[index] * vertices[index][1] for index in range(3)) / denominator
        return x, y


State_viewer = FtoStateViewer
