#!/usr/bin/env python
# coding: utf-8

import os
import sys
import timeit


import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Union
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as SSIM

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MODULE_DIR)

from utils import resize_image


class Profiler:
    def __init__(
        self,
        graph: tf.keras.Model,
        last_conv_layer: str,
        repeat: int = 150,
        logger=None,
    ):
        self.graph = graph
        self.repeat = repeat
        self.last_conv_idx = graph.layers.index(
            graph.get_layer(last_conv_layer)
        )
        self.logger = logger

    def prepare_profiling_dataset(self, imgs: Union[np.ndarray, tf.Tensor]):
        return imgs

    def feed_forward_subgraph(self, img: tf.Tensor, cut_layer_idx: int):
        """Feed forward tensor into subgraph

        Parameters
        ----------
        img : tf.Tensor
            2D array image
        cut_layer_idx : int
            layer index

        Returns
        -------
        [tf.Tensor]

        """
        subgraph = tf.keras.Sequential()
        for layer in self.graph.layers[:cut_layer_idx]:
            subgraph.add(layer)

        if len(img.shape) <= 3:
            return subgraph(img[np.newaxis])
        return subgraph(img)

    def get_SSIM_from_layer(self, image: tf.Tensor, layer_index: int):
        """Calculate SSIM of x from start layer to given index of layer

        Parameters
        ----------
        image (tf.Tensor): One color image (3D array: W, H, Channel).
            with [0, 255] scaled.
        layer_index (int): layer index.

        Returns
        -------
        SSIM(float): SSIM score

        """

        if image.ndim == 3:
            image = tf.identity(image[tf.newaxis])

        if image.ndim < 4:
            raise ValueError(
                (
                    "X image shape must be under 4 dims"
                    f"but given shape : {image.shape[1:]}"
                )
            )

        if isinstance(image, tf.Tensor):
            original_img = image.numpy()[0]
        elif isinstance(image, np.ndarray):
            original_img = image.copy()[0]  # 4 dims

        if self.logger:
            self.logger.info(f"In process (Profiler): {original_img.shape}")
        scaler = MinMaxScaler(feature_range=(0, 255))

        # Feedforwarding
        procesed_img = self.feed_forward_subgraph(image, layer_index)  # tesnor
        procesed_img = procesed_img.numpy().reshape(procesed_img.shape[1:])
        procesed_img = resize_image(
            procesed_img, reference_img=original_img, n_channel=1
        )

        # Gray scale
        procesed_img = procesed_img.mean(axis=-1)
        procesed_img = scaler.fit_transform(procesed_img)  # range from 0 to 255
        gray_original_img = original_img.mean(axis=-1)

        # SSIM
        print(gray_original_img.shape, procesed_img.shape)
        return SSIM(
            gray_original_img.astype(np.float16),
            procesed_img.astype(np.float16),
        )

    def run_privacy_profiling(
        self, images: Union[tf.Tensor, np.ndarray], return_df=True
    ):
        """Run privacy profile

        Parameters
        ----------
        batch_size: int


        Returns
        -------
        pd.DataFrame: return_df is true

        """
        if isinstance(images, np.ndarray):
            images = tf.convert_to_tensor(images)

        total_ssims = np.ones(shape=(len(images), self.last_conv_idx))
        for batch_idx, image in enumerate(images):
            for i in range(1, self.last_conv_idx):
                ssim = self.get_SSIM_from_layer(image=image, layer_index=i)
                total_ssims[batch_idx, i] = ssim

        if return_df:
            layer_names = [
                layer.name
                for layer in self.graph.layers[1 : self.last_conv_idx]
            ]

            df = pd.DataFrame(total_ssims, columns=layer_names)
            df = pd.DataFrame(df.unstack()).reset_index()
            df.columns = ["layer", "n", "ssim"]
            return df

        return total_ssims

    def get_layer_runtime(self, batch_size=100, repeat=150, return_size=True):
        """Get runtiem of each layer

        Parameters
        ----------
        batch_size: positive int.
        repeate: positive int.

        Return
        ------
        runtime: [sec, sec, sec...]
        dsize: /kbyte
        """

        xs = self.create_tensor_on_batch(batch_size)

        mem_sizes = []
        layer_runtimes = []

        for index in range(1, len(self.graph.layers)):
            front_model = tf.keras.Model(
                self.graph.input, self.graph.layers[index - 1].output
            )
            front_output = front_model(xs)
            mem_size = front_output.numpy().nbytes / 1024  # memory size
            mem_sizes.append(mem_size)

            runtimes = []
            for _ in range(repeat):
                start_time = timeit.default_timer()
                self.graph.layers[index](front_output)
                end_time = timeit.default_timer()

                runtime = end_time - start_time
                runtimes.append(runtime)

            layer_runtimes.append(runtimes)

        if return_size == False:
            return layer_runtimes
        else:
            return layer_runtimes, mem_sizes
