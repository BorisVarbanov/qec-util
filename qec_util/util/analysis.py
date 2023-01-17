from typing import Union

import numpy as np
from numpy.typing import NDArray


def error_prob(predictions: NDArray, values: NDArray) -> float:
    return np.mean(predictions ^ values)


def error_prob_decay(
    qec_round: Union[int, NDArray], error_rate: float
) -> Union[int, NDArray]:
    return 0.5 * (1 + (1 - 2 * error_rate) ** qec_round)
