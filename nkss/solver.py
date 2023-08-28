import numpy as np


def newton_raphson(
        func,
        j_func,
        init_states: np.ndarray,
        inputs: np.ndarray,
        error_min=1e-3,
        clip_lim=1e-1,
        iter_max=16384):
    """
    Newton-Raphson iteration

    :param func: function of states
    :param j_func: jacobian of func
    :param init_states: initial state values
    :param inputs: additional argument for func and j_func
    :param error_min: minimum error tolerance
    :param clip_lim: clipping to prevent gradient explosion
    :param iter_max: max iteration
    :return: final states
    """
    temp_states = init_states.copy()

    max_error = 1
    iter_num = 0
    while max_error > error_min :
        f = func(temp_states, inputs)
        df = j_func(temp_states, inputs)

        error = np.dot(np.linalg.inv(df), f)
        error = np.clip(error, -clip_lim, clip_lim)

        temp_states = np.subtract(temp_states, error)

        max_error = np.amax(np.absolute(error))
        iter_num += 1
    return temp_states
