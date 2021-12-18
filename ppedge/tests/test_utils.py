import os
import sys
import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PPEDGE_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PPEDGE_DIR)

from utils import resize_image


@pytest.mark.parametrize(
    "image_shape, shape, expected_shape",
    [
        pytest.param((240, 320, 64), (480, 640), (480, 640, 64)),
        pytest.param((240, 220, 1), (300, 600), (300, 600)),
        pytest.param((240, 220), (300, 600), (300, 600)),
    ],
)
def test_resize_image(image_shape, shape, expected_shape):
    image = np.ones(image_shape)

    result = resize_image(image, *shape)
    assert expected_shape == result.shape
