#!/usr/bin/env python
# coding: utf-8

import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

from ppedge.profiler import Profiler

def build_model(model='vgg'):

    if model == 'vgg':
        vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3), classes=10)
        vgg19.trainable = False
    
    # ADD Batch normalization
    base_model = tf.keras.models.Sequential()
    for layer in vgg19.layers:
        
        if 'conv1' in layer.name:
            base_model.add(layer)
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
            base_model.add(tf.keras.layers.BatchNormalization())
        elif 'conv2' in layer.name:
            base_model.add(layer)
            base_model.add(tf.keras.layers.BatchNormalization())
        else:
            base_model.add(layer)
    
    x = base_model.output
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    model = tf.keras.models.Model(base_model.input, x)
    
    return model



if __name__ == '__main__':
       
    # Graph building
    G = build_model()
    G.load_weights('./exp/stateFarm/cache/weight.h5')
    layer_names = [layer.name for layer in G.layers]
    
    
    # Model load
    files = []
    subdir = os.listdir('./exp/stateFarm/imgs/train/')
    for sub in subdir:

        sub_files = os.listdir('./exp/stateFarm/imgs/val/'+sub)
        sub_files = random.sample(sub_files, 50)
        PIL_imgs = [Image.open('./exp/stateFarm/imgs/val/'+sub+'/'+file) for file in sub_files]
        imgs = [np.array(img.resize((256, 256))) for img in PIL_imgs]  
        files.append(np.stack(imgs))

    imgs = np.stack(files).reshape(-1, 256, 256, 3)

    # Profiling
    profiler = Profiler(G, imgs, last_conv_layer='block5_pool')
    ssim_df = profiler.get_privacy_profiling(batch_size=30, return_df=True).reset_index()
    runtimes, memsize = profiler.get_layer_runtime()


    # To pd.DataFrame
    runtimes = pd.DataFrame(runtimes).T
    runtimes.columns = layer_names[1:]

    runtimes = pd.DataFrame(runtimes.stack()).reset_index()
    runtimes.columns = ['n', 'layer', 'latency']

    mem_df = pd.DataFrame(memsize)
    mem_df['layer'] = layer_names[1:]
    mem_df.columns = ['d_size', 'layer']
    
    
    # Save
    runtimes.to_csv('./result/server_runtime.csv')
    mem_df.to_csv('./result/data_size.csv')
    
    print("finished")
