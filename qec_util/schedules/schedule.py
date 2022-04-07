import imp
from typing import Dict
from itertools import groupby

import yaml

from quantumsim import Circuit


class Schedule:
    """
    Schedule class that allows the generation of
    quantumsim and stim circuits given a gate
    schedule.
    """

    def __init__(self, gate_schedule: Dict) -> None:
        if not isinstance(gate_schedule, dict):
            raise ValueError(
                f"schedule_setup expected as dict, instead got {type(gate_schedule)}"
            )
        self.name = gate_schedule.get("name", "")
        self.description = gate_schedule.get("description", "")

        self._gates = []

    def get_ciruit(self, layout, model, *, finalize=False, add_idling=False):
        """
        circuit_from_schedule Generates a quantumsim circuit from a schedule.

        Parameters
        ----------
        schedule : dict
            The circuit schedule
        layout : Layout
            The chip layout
        model : Model
            The quantum gate and error model.
        finalize : bool, optional
            Whether to insert idling gates and compile the circuit, by default False
        add_idling : bool, optional
            Whether to include idling gates idling gates for the circuit, by default False

        Returns
        -------
        Circuit
            The compile quantumsim.Circuit object.

        Raises
        ------
        ValueError
            If one of the qubits acted on by an operation is not in the layout.
        ValueError
            TODO: finish
        ValueError
            TODO: finish
        ValueError
            TODO: finish
        ValueError
            TODO: finish
        ValueError
            TODO: finish
        """

        gates = []
        qubits = layout.get_qubits()
        qubit_times = {qubit: 0.0 for qubit in qubits}
        circ_qubits = set()

        for layer in schedule["layers"]:
            time_start = layer["time_start"]

            for gate_dict in layer["gates"]:
                gate_label = gate_dict["label"]
                gate_method = getattr(model, gate_label)
                n_qubits = gate_dict["num_qubits"]

                if n_qubits == 1:
                    for qubit in gate_dict["qubits"]:
                        if qubit not in qubits:
                            raise ValueError(f"Qubit {qubit} not in layout.")

                        if qubit_times[qubit] > time_start:
                            raise ValueError(
                                f"Gate {gate_label} executed before previous operation has finished."
                            )
                        circ_qubits.add(qubit)

                        gate = gate_method(qubit, **gate_dict["parameters"])
                        gate = gate.shift(time_start=time_start)
                        gates.append(gate)

                        qubit_times[qubit] = gate.time_end

                elif n_qubits == 2:
                    for ctrl_q, target_q in gate_dict["qubits"]:
                        if ctrl_q not in qubits or target_q not in qubits:
                            raise ValueError(
                                f"Qubit(s) {ctrl_q}, {target_q}  not in layout."
                            )

                        if target_q not in layout.get_neighbors(ctrl_q):
                            raise ValueError(
                                f"Qubit {target_q} not coupled to {ctrl_q} in layout."
                            )

                        if (
                            qubit_times[ctrl_q] > time_start
                            or qubit_times[target_q] > time_start
                        ):
                            raise ValueError(
                                f"Gate {gate_label} executed before previous operation has finished."
                            )

                        circ_qubits.add(ctrl_q)
                        circ_qubits.add(target_q)

                        gate = gate_method(ctrl_q, target_q, **gate_dict["parameters"])
                        gate = gate.shift(time_start=time_start)
                        gates.append(gate)

                        qubit_times[ctrl_q] = gate.time_end
                        qubit_times[target_q] = gate.time_end

                else:
                    raise ValueError(
                        f"Unexpected number of qubits ({n_qubits}) for gate {gate_label}"
                    )

        circuit = Circuit(sorted(circ_qubits), gates)

        if add_idling:
            circuit = model.add_waiting_gates(circuit, time_end=schedule["circ_time"])

        if finalize:
            gen_basis = general(model.dim)
            bases_in = tuple(repeat(gen_basis, len(circ_qubits)))
            return circuit.finalize(bases_in=bases_in)

        return circuit


def clifford_layers(circuit: Circuit, gate_dict: Dict[str, str] = None):
    """
    clifford_layers Returns a list of clifford layers

    Parameters
    ----------
    circuit : quantumsim.Circuit
        _description_
    gate_dict : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    qubits = list(circuit.qubits)
    layers = []

    for _, layer_gates in groupby(circuit.gates, lambda gate: gate.time_start):
        gate_list = []
        for label, gates in groupby(layer_gates, lambda gate: gate.label):
            if label != "idle":
                qubits = [q for gate in gates for q in gate.qubits]
                if gate_dict:
                    gate_list.append((gate_dict[label], qubits))
                else:
                    gate_list.append((label, qubits))
        layers.append(gate_list)

    return layers
