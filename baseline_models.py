import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be

from utils import *
from data import *
from layers import *
from models import *

"""
############# Standard models #################
"""
input_shape=(28,28,1)
n_px = 28**2

### MLP models ###
def make_standard():
    model = keras.Sequential([layers.Input(input_shape),
                         layers.Flatten(),
                         layers.Dropout(0.25),
                         layers.Dense(100, activation="relu", use_bias=False),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid", use_bias=False)])
    
    model.compile(optimizer="rmsprop", 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def make_standard_dot():
    model = keras.Sequential([layers.Input(input_shape),
                         layers.Flatten(),
                         StaticLinear(n_px, n_px),
                         layers.Dropout(0.25),
                         layers.Dense(100, activation="relu", use_bias=False),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid", use_bias=False)])
    
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
                         layers.Dense(100, activation="relu", use_bias=False),
                         layers.Dropout(0.25),
                         layers.Dense(10, activation="sigmoid", use_bias=False)])
    
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

input_shape_cifar = (32,32,3)
n_px_cifar = (32*32*3)

def make_conv2D(weight_decay=1e-4, dropout_rate=0.25, n_hidden=1000):
    wdr=regularizers.l2(weight_decay)

    model = keras.Sequential([layers.Input(input_shape_cifar),
                            layers.BatchNormalization(),

                            layers.Conv2D(32, (3,3), kernel_regularizer=wdr, name="conv1", activation="relu"),
                            layers.Conv2D(32, (3,3), kernel_regularizer=wdr, name="conv2", activation="relu"),
                            layers.MaxPool2D((2,2), name="maxpool1"),
                            layers.Dropout(dropout_rate, name="dropout1"),

                            layers.Conv2D(64, (3,3), kernel_regularizer=wdr, name="conv3", activation="relu"),
                            layers.Conv2D(64, (3,3), kernel_regularizer=wdr, name="conv4", activation="relu"),
                            layers.MaxPool2D((2,2), name="maxpool2"),
                            layers.Dropout(dropout_rate, name="dropout2"),

                            layers.Flatten(name="flatten"),
                            layers.Dense(n_hidden, name="dense1", activation="relu"),
                            layers.Dropout(dropout_rate, name="dropout3"),
                            layers.Dense(10, name="dense2", activation="softmax"),

                            ])

    model.compile(optimizer="rmsprop", 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    return model

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