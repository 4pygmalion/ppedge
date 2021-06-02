#!/usr/bin/env python
# coding: utf-8

import timeit
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as SSIM

__all__ = ['Profiler']


def resize(img, reference_img, n_channel=3):
    '''

    Parameters
    ----------
    img: array
    reference_img: array with target image
    n_channel: int

    Returns
    -------
    img: array

    '''
    # Resizing
    if img.shape[0:2] != reference_img.shape[0:2]:
        img = cv2.resize(img, dsize=(reference_img.shape[0:2]))

    # channel wise padding
    if n_channel >= 2:
        img = np.stack([img for _ in range(n_channel)])
        img = np.transpose(img, axes=[1, 2, 0])

    return img


class Profiler(object):

    def __init__(self, graph, training_x, repeat=150):
        self.graph = graph
        self.repeat = repeat
        self.x = training_x
        self.scaler = MinMaxScaler(feature_range=(0, 255))


    def feed_forward_subgraph(self, img, cutoff):

        subgraph = tf.keras.Sequential()
        for layer in model.layers[:cutoff]:
            subgraph.add(layer)

        process_img = subgraph(img[np.newaxis])
        shape = process_img.shape[1:3]
        process_img = process_img.numpy().sum(axis=3).reshape(shape)

        return process_img


    def get_layer_runtime(self, batch_size):
        '''

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        pd.DataFrame
        '''


        front_model = tf.keras.Model(model.input, model.layers[index - 1].output)
        front_output = front_model(batch)

        runtimes = []
        for _ in range(repeat):
            start_time = timeit.default_timer()
            model.layers[index](front_output)
            end_time = timeit.default_timer()

            runtime = end_time - start_time
            runtimes.append(runtime)

        if return_size == False:
            return runtimes
        else:
            return_val = model.layers[index](front_output)
            mem_size = return_val.numpy().nbytes / 1024
            return runtimes, mem_size


    def get_privacy_profiling(self, ):

    def _get_privacy_img(self, x):
        '''

        Parameters
        ----------
        x: image

        Returns
        -------
        ssim : float32
        '''
        # randomly sample image
        original_img = x

        # get process image
        procesed_img = self.feed_forward_subgraph(x)
        procesed_img = resize(procesed_img, reference_img=original_img, n_channel=1)
        procesed_img = scaler.fit_transform(procesed_img)  # range from 0 to 255
        ori_gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        # SSIM
        ssim = SSIM(ori_gray_img, procesed_img)

        return ssim