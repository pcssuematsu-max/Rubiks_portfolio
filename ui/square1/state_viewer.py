"""Square-1-specific state viewer."""

import math
import tkinter as Tk


FACE_COLORS = {
    "U":"#EFEFEF",
    "D":"#BFBF00",
    "F":"#FF7F00",
    "R":"#0000BF",
    "B":"#7F0000",
    "L":"#007F00",
}
EDGE_SIDE_FACE = {
    "B":"F", "D":"R", "F":"B", "H":"L",
    "J":"F", "L":"R", "N":"B", "P":"L",
}
CORNER_SIDE_FALLBACK = {
    "A":("L", "F"), "C":("F", "R"), "E":("R", "B"), "G":("B", "L"),
    "I":("L", "F"), "K":("F", "R"), "M":("R", "B"), "O":("B", "L"),
}
PIECE_FACE = {
    "A":"U", "B":"U", "C":"U", "D":"U", "E":"U", "F":"U", "G":"U", "H":"U",
    "I":"D", "J":"D", "K":"D", "L":"D", "M":"D", "N":"D", "O":"D", "P":"D",
}


class Square1StateViewer(Tk.Canvas):
    def __init__(self, master, mini_mode = False):
        self.mini_mode = mini_mode
        self.size = 180 if mini_mode else 300
        Tk.Canvas.__init__(self, master, relief = Tk.RAISED, bd = 2 if mini_mode else 4, width = self.size, height = self.size, bg = "#BFBFBF")

    def set_color(self, state):
        self.delete("all")
        top = list(state[:12])
        bottom = list(state[12:24])
        top_side, bottom_side = self._side_states(top, bottom)
        radius = self.size * (0.145 if self.mini_mode else 0.15)
        self._draw_layer(top, top_side, self.size * 0.5, self.size * 0.21, radius)
        self._draw_middle_square(
            self._middle_regions(top, bottom),
            self.size * 0.5,
            self.size * 0.50,
            self.size * (0.35 if self.mini_mode else 0.32),
        )
        self._draw_layer(bottom, bottom_side, self.size * 0.5, self.size * 0.80, radius)

    def _draw_layer(self, slots, side_slots, cx, cy, radius):
        outer_width = 7 if self.mini_mode else 10
        for index, color_key in enumerate(slots):
            start = self._tk_arc_start(index)
            extent = 30
            fill = FACE_COLORS.get(PIECE_FACE.get(color_key), "#777777")
            self.create_arc(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                start = start,
                extent = extent,
                fill = fill,
                outline = fill,
                width = 0,
            )
            side_fill = FACE_COLORS.get(side_slots[index], "#777777")
            self.create_arc(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                start = start,
                extent = extent,
                style = Tk.ARC,
                outline = side_fill,
                width = outer_width,
            )
        self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline = "#222222", width = 1)
        self._draw_boundaries(slots, cx, cy, radius)
        self.create_oval(cx - radius * 0.18, cy - radius * 0.18, cx + radius * 0.18, cy + radius * 0.18, fill = "#BFBFBF", outline = "#222222")
        if not self.mini_mode:
            for index, color_key in enumerate(slots):
                x, y = self._point_at_clock_angle(cx, cy, radius * 0.68, 30 + index * 30)
                self.create_text(x, y, text = color_key, font = ("Arial", 9), fill = "#111111")

    def _side_states(self, top, bottom):
        cube = getattr(self.master, "cube", None)
        if cube is not None and hasattr(cube, "top_side_state") and hasattr(cube, "bottom_side_state"):
            return (
                self._merge_corner_side_slots(top, list(cube.top_side_state)),
                self._merge_corner_side_slots(bottom, list(cube.bottom_side_state)),
            )
        return (
            self._merge_corner_side_slots(top, [self._slot_side_face(top, index) for index in range(len(top))]),
            self._merge_corner_side_slots(bottom, [self._slot_side_face(bottom, index) for index in range(len(bottom))]),
        )

    def _merge_corner_side_slots(self, slots, side_slots):
        merged = list(side_slots)
        for index, piece in enumerate(slots):
            next_index = (index + 1) % len(slots)
            if piece == slots[next_index]:
                merged[next_index] = merged[index]
        return merged

    def _middle_regions(self, top, bottom):
        cube = getattr(self.master, "cube", None)
        if cube is not None and hasattr(cube, "middle_regions"):
            return cube.middle_regions
        return {
            "left":("F", "L", "L", "L", "B", "B"),
            "right":("B", "R", "R", "R", "F", "F"),
        }

    def _draw_middle_square(self, regions, cx, cy, size):
        half = size * 0.5
        slash_dx = math.tan(math.radians(15)) * half
        top_point = (cx + slash_dx, cy - half)
        bottom_point = (cx - slash_dx, cy + half)
        corners = {
            "tl":(cx - half, cy - half),
            "tr":(cx + half, cy - half),
            "br":(cx + half, cy + half),
            "bl":(cx - half, cy + half),
        }
        left_points = [corners["tl"], top_point, bottom_point, corners["bl"]]
        right_points = [top_point, corners["tr"], corners["br"], bottom_point]
        self._draw_middle_region_slots(regions["left"], left_points)
        self._draw_middle_region_slots(regions["right"], right_points)
        self.create_line(top_point[0], top_point[1], bottom_point[0], bottom_point[1], fill = "#111111", width = 4 if self.mini_mode else 5)
        self.create_rectangle(cx - half, cy - half, cx + half, cy + half, outline = "#111111", width = 2)

    def _draw_middle_region_slots(self, slots, points):
        if not slots:
            return
        slot_count = len(slots)
        for index, _ in enumerate(slots):
            left_t = index / slot_count
            right_t = (index + 1) / slot_count
            slot_points = [
                self._interpolate_point(points[0], points[1], left_t),
                self._interpolate_point(points[0], points[1], right_t),
                self._interpolate_point(points[3], points[2], right_t),
                self._interpolate_point(points[3], points[2], left_t),
            ]
            self.create_polygon(
                self._flatten_points(slot_points),
                fill = FACE_COLORS.get(self._middle_slot_face(slots, index), "#777777"),
                outline = "#222222",
                width = 1,
            )

    def _interpolate_point(self, start, end, ratio):
        return (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )

    def _flatten_points(self, points):
        flattened = []
        for x, y in points:
            flattened += [x, y]
        return flattened

    def _draw_boundaries(self, slots, cx, cy, radius):
        for boundary in range(12):
            if boundary not in (0, 6) and slots[boundary - 1] == slots[boundary]:
                continue
            x, y = self._point_at_clock_angle(cx, cy, radius, 15 + boundary * 30)
            width = 4 if boundary in (0, 6) else 1
            self.create_line(cx, cy, x, y, fill = "#111111", width = width)

    def _tk_arc_start(self, index):
        clock_angle = 15 + index * 30
        return 90 - (clock_angle + 30)

    def _point_at_clock_angle(self, cx, cy, radius, degrees):
        angle = math.radians(degrees)
        return cx + math.sin(angle) * radius, cy - math.cos(angle) * radius

    def _slot_side_face(self, slots, index):
        piece = slots[index]
        if piece in FACE_COLORS:
            return piece
        if piece in EDGE_SIDE_FACE:
            return EDGE_SIDE_FACE[piece]
        slot_count = len(slots)
        previous_piece = slots[(index - 1) % slot_count]
        next_piece = slots[(index + 1) % slot_count]
        fallback = CORNER_SIDE_FALLBACK.get(piece, ("F", "R"))
        if next_piece == piece:
            return EDGE_SIDE_FACE.get(previous_piece, fallback[0])
        if previous_piece == piece:
            return EDGE_SIDE_FACE.get(next_piece, fallback[1])
        return fallback[0]

    def _middle_slot_face(self, slots, index):
        slot = slots[index]
        if slot in FACE_COLORS:
            return slot
        return self._slot_side_face(slots, index)


State_viewer = Square1StateViewer
