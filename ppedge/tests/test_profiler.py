import os
import sys
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(os.path.dirname(TEST_DIR))

sys.path.append(MODULE_DIR)
from ppedge.profiler import Profiler


@pytest.fixture(scope="module")
def profiler():
    return Profiler()


def test_regress_fc_layer(profiler):
    expected_shape = (200, 100)
    assert expected_shape == profiler.regress_fc_layer(tuple(100)).shape
