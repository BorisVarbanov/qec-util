"""Module implementing layout generator for the surface code."""
from collections import defaultdict
from functools import partial
from itertools import count, cycle, product
from typing import Dict, Tuple

from .layout import Layout


def get_data_index(row: int, col: int, distance: int, start_ind: int = 1) -> int:
    """
    get_data_index Converts row and column to data qubit index for a distance d code. Assumes the initial index is 1 (as opposed to 0).

    Parameters
    ----------
    row : int
        The row of the data qubit.
    col : int
        The column of the data qubit.
    distance : int
        The distance of the code.
    start_ind : int, optional
        The starting index of the bottom-left data qubit, by default 1

    Returns
    -------
    int
        The index of the data qubit.
    """
    row_ind = row // 2
    col_ind = col // 2
    index = start_ind + (row_ind * distance) + col_ind
    return index


def shift_direction(row_shift: int, col_shift: int) -> str:
    """
    shift_direction Translates a row and column shift to a direction.

    Parameters
    ----------
    row_shift : int
        The row shift.
    col_shift : int
        The column shift.

    Returns
    -------
    str
        The direction.
    """
    ver_direction = "north" if row_shift > 0 else "south"
    hor_direction = "east" if col_shift > 0 else "west"
    direction = f"{ver_direction}_{hor_direction}"
    return direction


def invert_shift(row_shift: int, col_shift: int) -> Tuple[int, int]:
    """
    invert_shift Inverts a row and column shift.

    Parameters
    ----------
    row_shift : int
        The row shift.
    col_shift : int
        The column shift.

    Returns
    -------
    Tuple[int, int]
        The inverted row and column shift.
    """
    return -row_shift, -col_shift


def is_valid(row: int, col: int, max_size: int) -> bool:
    """
    is_valid Checks if a row and column are valid for a grid of a given size.

    Parameters
    ----------
    row : int
        The row.
    col : int
        The column.
    max_size : int
        The size of the grid.

    Returns
    -------
    bool
        Whether the row and column are valid.
    """
    if not 0 <= row < max_size:
        return False
    if not 0 <= col < max_size:
        return False
    return True


def add_missing_neighbours(neighbor_data: Dict) -> None:
    """
    add_missing_neighbours Adds None for missing neighbours in the neighbor data. Note that this modifies the dictionary in place.

    Parameters
    ----------
    neighbor_data : Dict
        The neighbor data - a dictionary that contains information about the layout connectivity (defined on a square grid, connection run diagonally).
    """
    directions = ["north_east", "north_west", "south_east", "south_west"]
    for neighbors in neighbor_data.values():
        for direction in directions:
            if direction not in neighbors:
                neighbors[direction] = None


def rot_surf_code(distance: int) -> Layout:
    """
    rot_surf_code Generates a rotated surface code layout.

    Parameters
    ----------
    distance : int
        The distance of the code.

    Returns
    -------
    Layout
        The layout of the code.
    """
    _check_distance(distance)  # Check if distance is an integer and is odd and positive

    name = f"Rotated d-{distance} surface code layout."  # Default name of the layout
    description = None  # No description

    freq_order = ["low", "mid", "high"]  # Frequencies, not useful in stab simulations

    int_order = dict(
        x_type=["north_east", "north_west", "south_east", "south_west"],
        z_type=["north_east", "south_east", "north_west", "south_west"],
    )  # Interaction orders of the checks

    layout_setup = dict(
        name=name,
        description=description,
        distance=distance,
        freq_order=freq_order,
        interaction_order=int_order,
    )  # Layout setup

    grid_size = 2 * distance + 1  # Grid size
    data_indexer = partial(
        get_data_index, distance=distance, start_ind=1
    )  # Indexer for data qubits
    valid_coord = partial(
        is_valid, max_size=grid_size
    )  # Check if a coordinate is valid

    pos_shifts = (1, -1)  # Possible shifts
    nbr_shifts = tuple(product(pos_shifts, repeat=2))  # Neighbour shifts

    layout_data = []  # Layout data
    neighbor_data = defaultdict(dict)  # Neighbour data dictionary

    freq_seq = cycle(("low", "high"))  # Frequency sequence

    # Add the data quibts - basically data qubits are in a grid over the odd rows/columns of the code layout
    for row in range(1, grid_size, 2):
        freq_group = next(freq_seq)
        for col in range(1, grid_size, 2):
            index = data_indexer(row, col)

            qubit_info = dict(
                qubit=f"D{index}",
                role="data",
                coords=[row, col],
                freq_group=freq_group,
                stab_type=None,
            )
            layout_data.append(qubit_info)

    # Add the x-type ancilla qubits. Ancilla qubits take up even rows/columns.
    x_index = count(1)
    for row in range(0, grid_size, 2):
        for col in range(2 + row % 4, grid_size - 1, 4):
            anc_qubit = f"X{next(x_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[row, col],
                freq_group="mid",
                stab_type="x_type",
            )
            layout_data.append(qubit_info)

            # Add data qubit neighbors and vice-versa
            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    # Add the z-type ancilla qubits. Ancilla qubits take up even rows/columns.
    z_index = count(1)
    for row in range(2, grid_size - 1, 2):
        for col in range(row % 4, grid_size, 4):
            anc_qubit = f"Z{next(z_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[row, col],
                freq_group="mid",
                stab_type="z_type",
            )
            layout_data.append(qubit_info)

            # Add data qubit neighbors and vice-versa
            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    # Fill in missing neighbours with None
    add_missing_neighbours(neighbor_data)

    # Add the neighbours to the layout data
    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)  # Create the layout
    return layout


def _check_distance(distance: int) -> None:
    """
    _check_distance Checks if the distance is valid.

    Parameters
    ----------
    distance : int
        The distance of the code.

    Raises
    ------
    ValueError
        If the distance is not an odd positive integer.
    ValueError
        If the distance is not an integer.
    """
    if not isinstance(distance, int):
        raise ValueError("distance provided must be an integer")
    if distance < 0 or (distance % 2) == 0:
        raise ValueError("distance must be an odd positive integer")
