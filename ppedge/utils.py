#!/usr/bin/env python
# coding: utf-8

import os
from re import DEBUG
import yaml
import logging
from cv2 import resize
from numpy import stack, transpose
from numpy import ndarray

PPEDGE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_logger(name, file_path=None):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    if file_path:
        file_handler = logging.FileHandler(file_path)
    else:
        file_handler = logging.FileHandler(
            os.path.join(PPEDGE_DIR, "logs/mainlog.txt")
        )

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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
    img (np.ndarray): 3dim (N, N, channel)
    height (int):
    width (int):

    Returns
    -------
    img: np.ndarray. 3dims

    """

    # In case of difference in W, and H

    return resize(img, dsize=(width, height))
