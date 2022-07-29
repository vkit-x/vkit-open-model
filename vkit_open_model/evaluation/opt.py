import math
import numpy as np


def pad_length_to_make_divisible(length: int, downsampling_factor: int):
    padded_length = math.ceil(length / downsampling_factor) * downsampling_factor
    return padded_length, padded_length - length


def pad_mat_to_make_divisible(
    # (H, W, *).
    mat: np.ndarray,
    downsampling_factor: int,
):
    height, width = mat.shape[:2]
    height, height_pad = pad_length_to_make_divisible(height, downsampling_factor)
    width, width_pad = pad_length_to_make_divisible(width, downsampling_factor)

    if height_pad == 0 and width_pad == 0:
        # No need to pad.
        return mat

    padded_shape = list(mat.shape)
    padded_shape[0] = height
    padded_shape[1] = width

    padded_mat = np.zeros(padded_shape, dtype=mat.dtype)
    padded_mat[:height - height_pad, :width - width_pad] = mat

    return padded_mat
