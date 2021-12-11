#!/usr/bin/env python
# coding: utf-8


from cv2 import resize
from numpy import stack, transpose
from numpy import ndarray


def resize_image(
    img: ndarray, reference_img: ndarray, n_channel: int = 3
) -> ndarray:

    """Resize image shape to reference_image shape

    Parameters
    ----------
    img (np.ndarray): 3dim (N, N, channel)
    reference_img: array with target image
    n_channel: int


    Returns
    -------
    img: np.ndarray. 3dims

    """
    if len(img.shape) != 3 or len(reference_img.shape) != 3:
        msg = f"image shape must be 3 dim, however, given: {img.shape}"
        raise ValueError(msg)

    # Resizing
    if img.shape[0:2] != reference_img.shape[0:2]:
        img = resize(img, dsize=(reference_img.shape[0:2]))

    # channel wise padding
    if n_channel >= 2:
        img = stack([img for _ in range(n_channel)])
        img = transpose(img, axes=[1, 2, 0])
    return img
