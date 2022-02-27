from typing import Optional, List
import re

import matplotlib.pyplot as plt
import matplotlib.patches as pt

from ...layouts import Layout
from quantumsim.circuits import Circuit


def plot(
    circuit: Circuit,
    layout: Layout,
    *,
    ax: Optional[plt.Axes] = None,
    qubit_order: Optional[List[str]] = None,
):
    """
    plot Plots the circuit

    Parameters
    ----------
    circuit : quantumsim.Circuit
        The quantum circuit to be plotted
    layout : qec_util.Layout
        The layout of the device.
    ax : Optional[plt.Axes], optional
        The axis on which the figure is plotted, by default None
    qubit_order : Optional[List[str]], optional
        The list of qubit labels that defines the order in the plot, by default None

    Returns
    -------
    matplotlib.Figure
        The figure with the plot.
    """
    plotter = MatplotlibPlotter(circuit, layout, ax, qubit_order=qubit_order)
    return plotter.plot()


class MatplotlibPlotter:
    """
    Matplotlib Plotter
    """

    zorders = {
        "line": 1,
        "marker": 1,
        "circle": 5,
        "box": 10,
        "text": 20,
    }

    def __init__(
        self,
        circuit: Circuit,
        layout: Layout,
        ax: Optional[plt.Axes] = None,
        *,
        qubit_order: Optional[List[str]] = None,
        unit_time: Optional[float] = 20.0,
    ):
        self.layout = layout
        self.gates = circuit.gates
        self.qubits = qubit_order or list(circuit.qubits)
        self._unit_time = unit_time

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=(7, len(circuit.qubits)))

        y_pad = 1
        x_pad = 1
        self.ax.set_ylim(-y_pad, len(self.qubits) + y_pad)
        self.ax.set_xlim(
            -x_pad,
            circuit.time_end / self._unit_time + x_pad,
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def plot(self):
        """
        plot Plots the circuit

        Returns
        -------
        Figure
            matplotlib figure.
        """
        for qubit in self.qubits:
            self._draw_qubit(qubit)
            self._annotate_qubit(qubit)
            self._plot_qubit_line(qubit)
        for gate in self.gates:
            self._plot_gate(gate)

        return self.fig

    def _plot_gate(self, gate):
        time_start = gate.time_start / self._unit_time
        duration = gate.duration / self._unit_time

        time = time_start + (0.5 * duration)

        if gate.label == "cphase":
            ctrl_q, target_q = gate.qubits
            q_start = self.qubits.index(ctrl_q)
            q_end = self.qubits.index(target_q)
            self.ax.plot(
                (time, time),
                (q_start, q_end),
                color="black",
                linewidth=1,
                zorder=self.zorders["circle"],
            )

            self.ax.add_patch(
                pt.Ellipse(
                    (time, q_start),
                    0.2,
                    0.2,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0,
                    zorder=self.zorders["circle"],
                )
            )
            self.ax.add_patch(
                pt.Ellipse(
                    (time, q_end),
                    0.2,
                    0.2,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0,
                    zorder=self.zorders["circle"],
                )
            )

        elif gate.label == "measure":
            ind = self.qubits.index(gate.qubits[0])
            rect = pt.Rectangle(
                (time_start, ind - 0.3),
                duration,
                0.6,
                linewidth=1,
                fc="white",
                ec="black",
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)

            self.ax.add_patch(
                pt.Arc(
                    (time + 0.1, ind - 0.1),
                    0.4,
                    0.3,
                    theta1=0,
                    theta2=180,
                    lw=1,
                    color="black",
                    zorder=self.zorders["text"],
                )
            )
            self.ax.arrow(
                time + 0.1,
                ind - 0.1,
                0.15,
                0.15,
                color="black",
                head_length=0.1,
                head_width=0.1,
                lw=0.5,
                zorder=self.zorders["text"],
            )
        elif gate.label == "rotate_y":
            inds = [self.qubits.index(q) for q in gate.qubits]
            angle = gate.params["angle"]
            gate_label = f"$Y_{{{angle}}}$"
            q_start, q_end = min(inds), max(inds)
            rect = pt.Rectangle(
                (time - 0.3, q_start - 0.3),
                0.5 * gate.duration / self._unit_time,
                q_end - q_start + 0.5,
                linewidth=1,
                fc="white",
                ec="black",
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)
            self.ax.text(
                time,
                q_start + 0.5 * (q_end - q_start),
                gate_label,
                ha="center",
                va="center",
                fontsize=6,
                zorder=self.zorders["text"],
            )
        elif gate.label == "rotate_x":
            inds = [self.qubits.index(q) for q in gate.qubits]
            angle = gate.params["angle"]
            gate_label = f"$X_{{{angle}}}$"

            q_start, q_end = min(inds), max(inds)
            rect = pt.Rectangle(
                (time - 0.3, q_start - 0.3),
                0.5 * gate.duration / self._unit_time,
                q_end - q_start + 0.5,
                linewidth=1,
                fc="white",
                ec="black",
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)
            self.ax.text(
                time,
                q_start + 0.5 * (q_end - q_start),
                gate_label,
                ha="center",
                va="center",
                fontsize=6,
                zorder=self.zorders["text"],
            )

        elif gate.label == "pipeline_measure":
            ind = self.qubits.index(gate.qubits[0])
            rect = pt.Rectangle(
                (time - 0.3, ind - 0.3),
                6.6,
                0.6,
                linewidth=1,
                fc="white",
                ec="black",
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)

            self.ax.add_patch(
                pt.Arc(
                    (time + 0.1, ind - 0.1),
                    0.4,
                    0.3,
                    theta1=0,
                    theta2=180,
                    lw=1,
                    color="black",
                    zorder=self.zorders["text"],
                )
            )
            self.ax.arrow(
                time + 0.1,
                ind - 0.1,
                0.15,
                0.15,
                color="black",
                head_length=0.1,
                head_width=0.1,
                lw=0.5,
                zorder=self.zorders["text"],
            )

            self.ax.plot(
                [time + 6.3, time + 6.3],
                [ind - 0.3, ind + 0.3],
                color="white",
                lw=2,
                zorder=self.zorders["box"] + 1,
            )

            n_points = 4
            x = time + 6.25
            y = ind - 0.3

            dy = 0.6 / (2 * n_points)
            dx = 0.1

            for i in range(n_points):
                cur_y = y + (2 * i) * dy
                self.ax.plot(
                    (x, x - dx),
                    (cur_y, cur_y + dy),
                    lw=1,
                    color="black",
                    zorder=self.zorders["box"] + 2,
                )
                self.ax.plot(
                    (x - dx, x),
                    (cur_y + dy, cur_y + 2 * dy),
                    lw=1,
                    color="black",
                    zorder=self.zorders["box"] + 2,
                )

        elif gate.label == "pipeline_dep":
            ind = self.qubits.index(gate.qubits[0])
            rect = pt.Rectangle(
                (time - 0.3, ind - 0.3),
                4.6,
                0.6,
                linewidth=1,
                fc="white",
                ec="gray",
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)

            self.ax.plot(
                [time - 0.3, time - 0.3],
                [ind - 0.3, ind + 0.3],
                color="white",
                lw=1.5,
                zorder=self.zorders["box"] + 1,
            )

            n_points = 4
            x = time - 0.25
            y = ind - 0.3

            dy = 0.6 / (2 * n_points)
            dx = 0.1

            for i in range(n_points):
                cur_y = y + (2 * i) * dy
                self.ax.plot(
                    (x, x + dx),
                    (cur_y, cur_y + dy),
                    lw=1,
                    color="gray",
                    zorder=self.zorders["box"] + 2,
                )
                self.ax.plot(
                    (x + dx, x),
                    (cur_y + dy, cur_y + 2 * dy),
                    lw=1,
                    color="gray",
                    zorder=self.zorders["box"] + 2,
                )

            self.ax.add_patch(
                pt.Arc(
                    (time + 0.3, ind - 0.1),
                    0.4,
                    0.3,
                    theta1=0,
                    theta2=180,
                    lw=1,
                    color="gray",
                    zorder=self.zorders["text"],
                )
            )
            self.ax.arrow(
                time + 0.3,
                ind - 0.1,
                0.15,
                0.15,
                color="gray",
                head_length=0.1,
                head_width=0.1,
                lw=0.5,
                zorder=self.zorders["text"],
            )

        else:
            raise ValueError("NOPE")

    def _draw_qubit(self, qubit):
        ind = self.qubits.index(qubit)
        metadata = {}
        if self.layout is not None:
            role = self.layout.param("role", qubit)
            if role == "data":
                freq_group = self.layout.param("freq_group", qubit)
                default_facecolor = "#ef5350" if freq_group == "high" else "#f48fb1"
            else:
                stab_type = self.layout.param("stab_type", qubit)
                default_facecolor = "#2196f3" if stab_type == "x_type" else "#4caf50"
        else:
            default_facecolor = "gray"
        circ = pt.Ellipse(
            (-0.6, ind),
            0.3,
            0.3,
            facecolor=metadata.get("facecolor", default_facecolor),
            edgecolor=metadata.get("edgecolor", "black"),
            linewidth=metadata.get("linewidth", 1),
            zorder=metadata.get("zorder", self.zorders["circle"]),
        )
        self.ax.add_patch(circ)

    def _annotate_qubit(self, qubit):
        n = self.qubits.index(qubit)
        self.ax.text(
            -1,
            n,
            self._latexify_label(str(qubit)),
            ha="center",
            va="center",
            fontsize=7,
            zorder=self.zorders["text"],
            weight="bold",
        )

    def _plot_qubit_line(self, qubit):
        n = self.qubits.index(qubit)
        self.ax.axhline(
            n, xmin=0.01, xmax=0.99, color="black", zorder=self.zorders["line"], lw=1
        )

    def _latexify_label(self, label: str):
        int_char = re.search("\d", label)
        if int_char is not None:
            start_ind = int_char.start()
            if re.search("\s", label[start_ind:]) is None:
                label = r"$" + label[:start_ind] + "_{" + label[start_ind:] + "}$"
        return label
