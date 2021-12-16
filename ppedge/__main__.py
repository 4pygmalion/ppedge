import os
import sys
import argparse
import random
import cv2
import numpy as np

import tensorflow as tf
from profile import Profiler
from models import build_model

PPEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PPEDGE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
sys.path.append(PPEDGE_DIR)

from utils import get_logger, open_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="server")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="the number of batch size",
        default=5,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    DEVICE = args.device
    BATCH_SIZE = args.batch_size

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # 텐서플로가 세 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as e:
            print(e)

    LOGGER = get_logger(name="Main Logger")
    LOGGER.info("-" * 30)
    LOGGER.info("In Process: Image Preparation")

    CONFIG = open_config(os.path.join(ROOT_DIR, "config.yaml"))
    TRAIN_PATH = os.path.join(DATA_DIR, CONFIG["DATASETS"]["STATEFARM"])
    FOLDERS = [
        os.path.join(TRAIN_PATH, folder) for folder in os.listdir(TRAIN_PATH)
    ]
    IMAGE_PATHS = [
        os.path.join(folder, img_path)
        for folder in FOLDERS
        for img_path in os.listdir(folder)
    ]
    PROFILING_IMGS = random.sample(IMAGE_PATHS, BATCH_SIZE)
    imgs = np.stack([cv2.imread(img) for img in PROFILING_IMGS])
    LOGGER.info(f"Iamge shape:{imgs.shape}")
    LOGGER.info("In Process: Model building")
    model = build_model()

    LOGGER.info("In Process: Device profiling")
    profiler = Profiler(model, "block5_conv4", logger=LOGGER)
    profiler.run_privacy_profiling(imgs)

    LOGGER.info("In Process: End of profiling")
