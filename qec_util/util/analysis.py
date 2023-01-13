from typing import Union

import numpy as np
from numpy.typing import NDArray


def accuracy(predictions: NDArray, values: NDArray) -> float:
    return np.mean(predictions ^ values)


def log_fidelity(
    qec_round: Union[int, NDArray], error_rate: float, round_offset: int = 0
) -> Union[int, NDArray]:
    return 0.5 * (1 + (1 - 2 * error_rate) ** (qec_round - round_offset))
