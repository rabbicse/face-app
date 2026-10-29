import numpy as np


def to_model_input(rgb_faces: np.ndarray) -> np.ndarray:
    """
    :param rgb_faces: uint8 array (H, W, 3) or (N, H, W, 3), RGB order
    :return: float32 NCHW tensor normalised to [-1, 1] with ``(x - 127.5) / 127.5``
    """
    arr = np.asarray(rgb_faces)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) RGB images, got {arr.shape}")
    x = (arr.astype(np.float32) - 127.5) / 127.5
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
