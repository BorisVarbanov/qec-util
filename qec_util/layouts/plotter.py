import re
from typing import Optional
from itertools import combinations

import matplotlib.pyplot as plt

RE_FILTER = re.compile("([a-zA-Z]+)([0-9]+)")


def plot(
    layout,
    label_qubits=True,
    *,
    axis: Optional[plt.Axes] = None,
):
    plotter = MatplotlibPlotter(layout, axis)
    return plotter.plot(label_qubits)


class MatplotlibPlotter:
    zorders = dict(
        line=3,
        patch=1,
        circle=5,
        text=20,
    )

    qubit_circ_params = dict(
        radius=0.35,
        lw=1,
        ec="black",
    )

    label_params = dict(
        ha="center",
        va="center",
        weight="bold",
    )

    line_params = dict(
        color="black",
        linestyle="--",
        lw=1,
    )

    patch_params = dict(lw=0, alpha=0.3)

    def __init__(
        self,
        layout,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        self.layout = layout

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self._qubit_cords = {}

    def _label_qubit(self, qubit, x, y):
        match = RE_FILTER.match(qubit)
        if match is None:
            raise ValueError(f"Unexpected qubit label {qubit}")
        label, ind = match.groups()
        q_label = f"${label}_\\mathrm{{{ind}}}$"
        self.ax.text(
            x,
            y,
            q_label,
            zorder=self.zorders["text"],
            **self.label_params,
        )

    def _draw_qubit_circ(self, x, y, color):
        qubit_circ = plt.Circle(
            (x, y),
            color=color,
            zorder=self.zorders["circle"],
            **self.qubit_circ_params,
        )
        self.ax.add_artist(qubit_circ)

    def _draw_patch(self, cords, color):
        patch = plt.Polygon(
            cords,
            color=color,
            zorder=self.zorders["patch"],
            **self.patch_params,
        )
        self.ax.add_artist(patch)

    def _draw_connection(self, x_cords, y_cords):
        self.ax.plot(x_cords, y_cords, **self.line_params)

    def _draw_qubits(self, label_qubits=True):
        qubits = self.layout.get_qubits()

        init_qubit = qubits.pop()
        init_cords = (0, 0)

        drawn_qubits = set()

        def _dfs_draw(qubit, x, y):
            if qubit not in drawn_qubits:
                role = self.layout.param("role", qubit)
                if role == "data":
                    freq_group = self.layout.param("freq_group", qubit)
                    color = "#d32f2f" if freq_group == "high" else "#e57373"
                else:
                    stab_type = self.layout.param("stab_type", qubit)
                    color = "#2196f3" if stab_type == "x_type" else "#4caf50"

                self._qubit_cords[qubit] = (x, y)
                self._draw_qubit_circ(x, y, color)
                if label_qubits:
                    self._label_qubit(qubit, x, y)
                drawn_qubits.add(qubit)

                neighbors_dict = self.layout.param("neighbors", qubit)
                for con_dir, neighbour in neighbors_dict.items():
                    if neighbour:
                        ver_dir, hor_dir = con_dir.split("_")

                        new_x = x - 1 if hor_dir == "west" else x + 1
                        new_y = y - 1 if ver_dir == "south" else y + 1

                        _dfs_draw(neighbour, new_x, new_y)
                        self._draw_connection((x, new_x), (y, new_y))

        _dfs_draw(init_qubit, *init_cords)
        qubit_cords = list(self._qubit_cords.values())
        x_cords, y_cords = zip(*qubit_cords)
        self.ax.set_xlim(min(x_cords) - 1, max(x_cords) + 1)
        self.ax.set_ylim(min(y_cords) - 1, max(y_cords) + 1)

    def _draw_patches(self):
        anc_qubits = self.layout.get_qubits(role="anc")
        for anc in anc_qubits:
            anc_cords = self._qubit_cords[anc]
            stab_type = self.layout.param("stab_type", anc)
            color = "#2196f3" if stab_type == "x_type" else "#4caf50"
            neigbors = self.layout.get_neighbors(anc)
            for data_pair in combinations(neigbors, 2):
                pair_cords = list(self._qubit_cords[data] for data in data_pair)
                dist = sum([abs(i - j) for i, j in zip(*pair_cords)])
                if dist <= 2:
                    cords = [anc_cords, *pair_cords]
                    self._draw_patch(cords, color)

    def plot(self, label_qubits=True, draw_patches=True):
        self._draw_qubits(label_qubits)
        if draw_patches:
            self._draw_patches()
        return self.fig
