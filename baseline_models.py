import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be

from utils import *
from data import *
from layers import *
from easy_model import *

"""
############# Standard models #################
"""
input_shape=(28,28,1)
n_px = 28**2

def make_standard():
    model = keras.Sequential([layers.Input(input_shape),
                         layers.Flatten(),
                         layers.Dropout(0.25),
                         layers.Dense(100, activation="relu"),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid")])
    
    model.compile(optimizer="rmsprop", 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def make_standard_dot():
    model = keras.Sequential([layers.Input(input_shape),
                         layers.Flatten(),
                         StaticLinear(n_px, n_px),
                         layers.Dropout(0.25),
                         layers.Dense(100, activation="relu"),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid")])
    
    direction = 2.0 * (tf.cast(tf.random.uniform((1, n_px)) > 0.5, dtype="float")) - 1.0
    model.layers[2].w = tf.multiply(tf.eye(n_px), direction)
    
    model.compile(optimizer="rmsprop", 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def make_standard_RP():
    model = keras.Sequential([layers.Input(input_shape),
                         layers.Flatten(),
                         StaticLinear(n_px, n_px, 100),
                         layers.BatchNormalization(),
                         layers.Dropout(0.25),
                         layers.Dense(100, activation="relu"),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid")])
    
    model.compile(optimizer="rmsprop", 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def test_standard(constructor, ds_train, ds_test, n_trials=10, n_epochs=2, n_batch=128):
    accs = []
    
    for i in range(n_trials):
        model = constructor()
        model.fit(ds_train, epochs=n_epochs, batch_size=n_batch)
        acc = model.evaluate(ds_test, batch_size=n_batch)
        accs.append(acc[1])
        
    return np.array(accs)

"""
############# Phasor models #################
"""

def make_phasor():
    model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="none")
    model.compile(optimizer="rmsprop")
    return model

def make_phasor_dot():
    model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="dot")
    model.compile(optimizer="rmsprop")
    return model

def make_phasor_NP():
    model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
    model.compile(optimizer="rmsprop")
    return model

def test_phasor(constructor, ds_train, ds_test, n_trials=10, n_epochs=2):
    accs = []
    
    for i in range(n_trials):
        model = constructor()
        model.train(ds_train, n_epochs)
        acc = model.accuracy(ds_test, confusion=False)
        accs.append(acc[0])
        
    return np.array(accs)