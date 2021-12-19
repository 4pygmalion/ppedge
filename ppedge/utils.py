#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import logging
from cv2 import resize
from numpy import ndarray

PPEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PPEDGE_DIR, "log")


def get_logger(name, file_path=None):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    if file_path:
        file_handler = logging.FileHandler(file_path)
    else:
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, "mainlog.txt"))

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def open_config(path):
    with open(path, "r") as file_handle:
        return yaml.load(file_handle, Loader=yaml.FullLoader)


def resize_image(img: ndarray, height: int, width: int) -> ndarray:

    """Resize image shape to reference_image shape

    Parameters
    ----------
    img (np.ndarray): One iamge. 3dim (N, N, channel)
    height (int):
    width (int):

    Returns
    -------
    img: np.ndarray. 3dims

    Example
    >>> image= np.ones(shape=(240, 220, 5))
    >>> result = resize_iamge(image, shape=(300, 600))
    >>> result.shape
    (300, 600, 5)
    """

    # In case of difference in W, and H

    return resize(img, dsize=(width, height))
