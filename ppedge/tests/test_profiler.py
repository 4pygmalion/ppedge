import os
import sys
import pytest
import tensorflow as tf
from unittest.mock import MagicMock

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(os.path.dirname(TEST_DIR))

sys.path.append(MODULE_DIR)
from ppedge.profile import Profiler


@pytest.fixture(scope="module")
def profiler():
    model = tf.keras.applications.VGG19(
        include_top=False, input_shape=(256, 256, 3), classes=10
    )
    return Profiler(graph=model)


def test_find_last_conv_idx(profiler):
    assert 20 == profiler._find_last_conv_idx()


# def test_regress_fc_layer(profiler):
#     expected_shape = (200, 100)
#     assert expected_shape == profiler.regress_fc_layer().shape
