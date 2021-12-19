import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(os.path.dirname(TEST_DIR))

sys.path.append(MODULE_DIR)
from ppedge.profile import Profiler


@pytest.fixture(scope="module")
def test_images():
    return tf.ones(shape=(5, 240, 360, 3))


@pytest.fixture(scope="module")
def profiler():
    model = tf.keras.applications.VGG19(
        include_top=False, input_shape=(256, 256, 3), classes=10
    )
    return Profiler(graph=model, logger=MagicMock())


def test_find_last_conv_idx(profiler):
    assert 20 == profiler._find_last_conv_idx()


def test_get_SSIM_from_layer(profiler, test_images):
    result_ssims = profiler.get_SSIM_from_layer(test_images, layer_index=3)
    assert isinstance(result_ssims, list)
    assert test_images.shape[0] == len(result_ssims)


def test_run_privacy_profiling(profiler, test_images):
    result = profiler.run_privacy_profiling(test_images, False)
    assert isinstance(result, list)
    assert profiler.last_conv_idx - 1 == len(result)
