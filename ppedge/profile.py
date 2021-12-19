#!/usr/bin/env python
# coding: utf-8

import os
import sys
import timeit


import numpy as np
from numpy.core import multiarray
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
        last_conv_layer: str = None,
        repeat: int = 150,
        logger=None,
    ):
        self.graph = graph
        self.repeat = repeat
        self.logger = logger
        self.last_conv_idx = self._find_last_conv_idx()

    def _find_last_conv_idx(self) -> int:

        last_idx = int()
        for layer_idx, layer in enumerate(self.graph.layers):
            if "conv" in layer.name and last_idx < layer_idx:
                last_idx = layer_idx

        return last_idx

    def feed_forward_subgraph(self, img: tf.Tensor, cut_layer_idx: int):
        """Feed forward tensor into subgraph

        Note:
            Some graph not sequential.

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

        model = tf.keras.models.Sequential()
        for layer in self.graph.layers[:cut_layer_idx]:
            model.add(layer)

        return model(img)

    def get_SSIM_from_layer(self, images: tf.Tensor, layer_index: int):
        """Calculate SSIM of x from start layer to given index of layer

        Parameters
        ----------
        images (tf.Tensor): color images (4D array: Batch, W, H, Channel).
            with [0, 255] scaled.
        layer_index (int): layer index.

        Returns
        -------
        SSIM(float): SSIM scores

        """
        if self.logger:
            self.logger.info(f"get_SSIM_from_layer: Input image shape{images.shape}")

        if isinstance(images, tf.Tensor):
            original_imgs = images.numpy()
        elif isinstance(images, np.ndarray):
            original_imgs = images.copy()

        if original_imgs.ndim >= 4 and original_imgs.shape[-1] >= 2:
            gray_original_imgs = original_imgs.mean(axis=-1)

        if self.logger:
            self.logger.debug(f"Original image: {original_imgs.shape}")
        scaler = MinMaxScaler(feature_range=(0, 255))

        # Feedforwarding
        procesed_imgs = self.feed_forward_subgraph(images, layer_index)  # tesnor
        self.logger.debug(f"Output image shape: {procesed_imgs.shape}")
        procesed_imgs = procesed_imgs.numpy()

        resize_imgs = []
        for img in procesed_imgs:
            height, width = images.shape[1], images.shape[2]
            resize_imgs.append(resize_image(img, height=height, width=width))
        resize_imgs = np.stack(resize_imgs)

        # Multichannel to gray channel
        if resize_imgs.shape[-1] >= 2:
            gray_procesed_imgs = resize_imgs.mean(axis=-1)

        gray_procesed_imgs = [scaler.fit_transform(img) for img in gray_procesed_imgs]

        # SSIM
        ssims = []
        for gray_origin_img, gray_procesed_imgs in zip(
            gray_original_imgs, gray_procesed_imgs
        ):
            ssim = SSIM(
                gray_origin_img.astype(np.float16),
                gray_procesed_imgs.astype(np.float16),
            )
            ssims.append(ssim)
        return ssims

    def run_privacy_profiling(self, images: Union[tf.Tensor, np.ndarray], return_df=True):
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

        total_ssims = []
        for layer_index in range(1, self.last_conv_idx):
            if self.logger:
                self.logger.info(
                    f"In process: run profiling of {layer_index} index layer"
                )
            ssims = self.get_SSIM_from_layer(images, layer_index=layer_index)
            total_ssims.append(ssims)

        if return_df:
            layer_names = [
                layer.name for layer in self.graph.layers[1 : self.last_conv_idx]
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
