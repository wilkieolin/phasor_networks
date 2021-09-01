"""
This file defines functions used to import and handle data. 

Wilkie Olin-Ammentorp, 2021
University of Califonia, San Diego
"""

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from utils import similarity

"""
Normalizes images: `uint8` -> `float32`.
"""
def normalize_img(image, label):
    
    return tf.cast(image, tf.float32) / 255., label

"""
Load a standard TF image dataset and apply the normal transforms
(cache, shuffle, batch, prefetch)
"""
def load_dataset(dataset, n_batch=-1,  normalize=True):
    (ds_train, ds_test), ds_info = tfds.load(dataset, 
                    split=['train', 'test'], 
                    data_dir="~/data",
                    shuffle_files=True,
                    as_supervised=True,
                    with_info=True)

    if normalize:
        ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

    if n_batch > 0:
        ds_train = ds_train.batch(n_batch)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if n_batch > 0:
        ds_test = ds_test.batch(n_batch)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_info


"""
Recreate the original dataset from the loader (only use on small datasets that fit in RAM)
"""
def get_raw_dat(data):
    data = [(d[0], d[1]) for d in iter(data)]
    xs = tf.concat([d[0] for d in data],axis=0)
    ys = tf.concat([d[1] for d in data],axis=0)
    
    return xs, ys

"""
Given a list of symbols x, select a subset (randomly, by default, otherwise the first n samples)
"""
def cut(x, n, rand_sel=True):
    d = []
    n_s = x[0].shape[0]

    if rand_sel:
        inds = tf.random.shuffle(tf.range(0,n_s))[0:n]
        
    for i in range(len(x)):
        if rand_sel:
            data = tf.gather(x[i], inds, axis=0)
            d.append(data)
        else:
            d.append(x[i][0:n,...])
        
    return tuple(d)

"""
Given a confusion matrix, calculate the corresponding accuracy score.
"""
def confusion_to_accuracy(confusion):
    total = tf.math.reduce_sum(confusion)
    correct = tf.math.reduce_sum(tf.linalg.diag_part(confusion))
    return correct / total