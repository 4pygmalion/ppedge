#!/usr/bin/env python
# coding: utf-8

import timeit
import random

# import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as SSIM

__all__ = ["Profiler", "resize"]


def resize(img, reference_img, n_channel=3):
    """

    Parameters
    ----------
    img: array.
        3dim (N, N, channel)
    reference_img: array with target image
    n_channel: int


    Returns
    -------
    img: np.ndarray. 3dims

    """
    if len(img.shape) != 3 or len(reference_img.shape) != 3:
        raise ValueError(
            "image shape must be 3 dim, however, input: {}".format(img.shape)
        )

    # Resizing
    if img.shape[0:2] != reference_img.shape[0:2]:
        img = cv2.resize(img, dsize=(reference_img.shape[0:2]))

    # channel wise padding
    if n_channel >= 2:
        img = np.stack([img for _ in range(n_channel)])
        img = np.transpose(img, axes=[1, 2, 0])
    return img


class Profiler(object):
    """

    Parameters
    ----------
    graph (tensorflow.keras.models.Model)
    training_x (np.ndarray)

    """

    def __init__(self):
        pass

    # def __init__(self, graph, train_x, last_conv_layer, repeat=150):

    # self.graph = graph
    # self.repeat = repeat
    # self.x = train_x
    # self.scaler = MinMaxScaler(feature_range=(0, 255))
    # self.last_conv_layer = last_conv_layer
    # self.last_conv_index = graph.layers.index(graph.get_layer(last_conv_layer))

    def regress_fc_layer(self, shape: tuple, batch=200) -> tf.Tensor:
        return tf.random.normal(shape=(batch, *shape))

    def _generate_input_tensor(self, input_shape: tuple) -> tf.tensor:
        return tf.random.normal(shape=input_shape)

    def do_profiling(
        self, layer: tf.keras.layers.Layer, **kwargs
    ) -> sklearn.linear_model.LinearRegression:

        return

    def _get_batch(self, batch_size):
        random_x = random.sample(range(len(self.x)), batch_size)
        return self.x[random_x].astype("float32")

    def feed_forward_subgraph(self, img, cutoff):

        subgraph = tf.keras.Sequential()
        for layer in self.graph.layers[:cutoff]:
            subgraph.add(layer)

        if len(img.shape) <= 3:
            process_img = subgraph(img[np.newaxis])
        else:
            process_img = subgraph(img)
        return process_img

    def _get_privacy_img(self, x, cutoff):
        """
        Parameters
        ----------
        x: image. not scaled.


        Returns
        -------
        ssim : float32
        """
        if len(x.shape) >= 4:
            raise ValueError(
                "X image shape must be under 4 dims, but given shape : {}".format(
                    x.shape
                )
            )

        # randomly sample image
        original_img = x.copy()

        # get process image
        procesed_img = self.feed_forward_subgraph(x / 255, cutoff)
        procesed_img = procesed_img.numpy().reshape(procesed_img.shape[1:])
        procesed_img = resize(procesed_img, reference_img=original_img, n_channel=1)
        procesed_img = procesed_img.sum(axis=-1)
        procesed_img = self.scaler.fit_transform(procesed_img)  # range from 0 to 255

        ori_gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        # SSIM
        return SSIM(ori_gray_img, procesed_img)

    def get_privacy_profiling(self, batch_size, return_df=True):
        """

        Parameters
        ----------
        batch_size: int


        Returns
        -------
        pd.DataFrame: return_df is true

        """
        xs = self._get_batch(batch_size)

        total_ssims = []
        for x in xs:
            x_ssims = []

            for p in range(1, self.last_conv_index):
                ssim = self._get_privacy_img(x, cutoff=p)
                x_ssims.append(ssim)

            total_ssims.append(x_ssims)

        if return_df:
            layer_names = [
                layer.name for layer in self.graph.layers[1 : self.last_conv_index]
            ]
            df = pd.DataFrame(total_ssims, columns=layer_names)
            df = pd.DataFrame(df.unstack()).reset_index()
            df.columns = ["layer", "n", "ssim"]
            return df

        else:
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

        xs = self._get_batch(batch_size)

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
