#!/usr/bin/env python
# coding: utf-8

import logging
import yaml
from cv2 import resize
from numpy import stack, transpose
from numpy import ndarray


def get_logger(name):
    logger = logging.getLogger(name=name)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def open_config(path):
    with open(path, "r") as file_handle:
        return yaml.load(file_handle, Loader=yaml.FullLoader)


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
    if img.ndim != 3 or reference_img.ndim != 3:
        msg = (
            f"image shape must be 3 dim, however"
            f"given(img, ref_img): {img.shape, reference_img.shape}"
        )
        raise ValueError(msg)

    # Resizing
    if img.shape[0:2] != reference_img.shape[0:2]:
        img = resize(img, dsize=(reference_img.shape[0:2]))

    # channel wise padding
    if n_channel >= 2:
        img = stack([img for _ in range(n_channel)])
        img = transpose(img, axes=[1, 2, 0])
    return img
