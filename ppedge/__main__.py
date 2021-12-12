import os
import sys
import argparse

import logging
from profile import Profiler
from models import build_model
# PPEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(PPEDGE_DIR)


# 서버 및 디바이스인지 알려주는 키워드가 필요.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="server")
    parser.add_argument("-b", "--batch_size", type=int, help="the number of batch size", default=5)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    DEVICE = args.device
    BATCH_SIZE = args.batch_size

    # LOGGING
    LOGGER = logging.getLogger(name="Main Logger")
    STREAM_HANLDER = logging.StreamHandler()
    FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    STREAM_HANLDER.setFormatter(FORMATTER)
    LOGGER.addHandler(STREAM_HANLDER)

    LOGGER.info("Model building")
    model = build_model()

    LOGGER.info("Device profiling")
    profiler = Profiler(graph=model, last_conv_layer="block5_conv4")
    profiler.run_privacy_profiling(batch_size=3)


    LOGGER.info("End of profiling")