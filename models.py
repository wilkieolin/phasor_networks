"""
This file contains the models used for deep phasor network experiments. 
These models define (1) the small MLP model used for MNIST-format tasks,
and (2) the larger model used on the CIFAR test set. 

Wilkie Olin-Ammentorp, 2021
University of Califonia, San Diego
"""

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be
from tensorflow.keras import regularizers

from data import get_raw_dat, cut
from layers import *
from utils import *
from scipy.stats import mode

"""
Converts a vector of features from a real-valued input, and projects/normalizes them into a VSA symbol.
"""
class VSA_Encoder(keras.Model):
    def __init__(self, n_in, n_out, overscan=1.0, sigma=3.0, name='VSA Encoder'):
        super(VSA_Encoder, self).__init__()
        self.transform = StaticLinear(n_in, n_out, overscan)
        self.norm_symbols = Normalize(sigma)

        
    def call(self, inputs, **kwargs):
        x = self.transform(inputs)
        x = self.norm_symbols(x, **kwargs)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x

    def reverse(self, inputs):
        x = self.norm_symbols.reverse(inputs)
        x = self.transform.reverse(x)
        
        return x    

"""
Simple multi-layer perceptron model using the phasor basis of activation.
"""
class PhasorModel(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(PhasorModel, self).__init__()
        #static parameters
        #number of neurons in the hidden layer
        self.n_hidden = kwargs.get("n_hidden", 100)
        self.n_classes = kwargs.get("n_classes", 10)
        #the 'DC' offset for all output neurons phase
        self.onehot_offset = kwargs.get("onehot_offset", 0.0)
        #the offset the active class will be pushed to (quadrature by default)
        self.onehot_phase = kwargs.get("onehot_phase", 0.5)
        #the projection to use for real / phasor conversion at the front-end
        self.projection = kwargs.get("projection", "dot")
        #the dropout rate between layers for training
        self.dropout_rate = kwargs.get("dropout_rate", 0.25)

        #dynamic parameters
        self.dyn_params = {
            #how fast potentials in the R&F neuron will decay (negative values cause decay)
            "leakage" : kwargs.get("leakage", -0.2),
            #the eigenperiod for the R&F neuron
            "period" : kwargs.get("period", 1.0),
            #the width of the box current produced by a spike
            "window" : kwargs.get("window", 0.05),
            #spike detection mode ('gradient' or 'cyclemax')
            "spk_mode" : kwargs.get("spk_mode", "gradient"),
            #imaginary (voltage) threshold of the R&F neuron
            "threshold" : kwargs.get("threshold", 0.03),
            #how long the dynamic network will be executed for
            "exec_time" : kwargs.get("exec_time", 10.0),
            #maximum dt for the differential solver
            "max_step" : kwargs.get("max_step", 0.02)
        }
        #how many cycles of spikes will be presented as a dynamic input
        self.repeats = kwargs.get("repeats", 10)

        n_feat = tf.reduce_prod(input_shape).numpy().astype('int')

        self.flatten = layers.Flatten(name="flatten")
        if self.projection == "NP":
            #overscan controls the sparsity of the random projection; lower is sparser
            self.overscan = kwargs.get("overscan", 100)
            self.sigma = kwargs.get("sigma", 3.0)
            self.image_encoder = VSA_Encoder(n_feat, n_feat, overscan=self.overscan, sigma=self.sigma)
        elif self.projection == "dot":
            self.direction = 2.0 * (tf.cast(tf.random.uniform((1, n_feat)) > 0.5, dtype="float")) - 1.0
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout1")
        self.dense1 = CmpxLinear(self.n_hidden, **self.dyn_params, name="complex1")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout2")
        self.dense2 = CmpxLinear(self.n_classes, **self.dyn_params, name="complex2")

    """
    Make a prediction of class by averaging across all the output cycles
    """
    def _mean_prediction(self, yh):
        #take the average across phases
        yh_avg = np.nanmean(yh, axis=1)
        yh_i = np.argmin(np.abs(yh_avg - self.onehot_phase), axis=1)
        return yh_i

    """
    Make a prediction of class by taking the mode of the output cycles
    """
    def _mode_prediction(self, yh):
        yh_i = np.argmin(np.abs(yh - self.onehot_phase), axis=2)
        yh_i = mode(yh_i, axis=1)[0]
        return yh_i.ravel()

    """
    Make a prediction of class using static execution (only 1 output to consider)
    """
    def _static_prediction(self, yh):
        yh_i = tf.argmin(tf.math.abs(yh - self.onehot_phase), axis=1)
        return yh_i

    """
    Measure the accuracy of the network on a test set using either exeuction method.
    Optionally produces a confusion matrix and similarity matrix. 
    """
    def accuracy(self, loader, confusion=True,  similarity=False, method="static"):
        guesses = np.zeros((self.n_classes, self.n_classes), dtype=np.int)

        if similarity:
            samples = np.zeros_like(guesses, dtype=np.float)

        for data in loader:
            x, y = data
            ns = x.shape[0]

            if method == "dynamic_mean" or method == "dynamic":
                yh = self.call_dynamic(x)
                yh_i = self._mean_prediction(yh)

            elif method == "dynamic_mode":
                yh = self.call_dynamic(x)
                yh_i = self._mode_prediction(yh)

            else:
                yh = self.call(x)
                yh_i = self._static_prediction(yh)

            for i in range(ns):
                guesses[y[i], yh_i[i]] += 1

                if similarity:
                    samples[y[i],:] += similarity(yh, self.to_phase(y))

        rvals = []
        total = tf.math.reduce_sum(guesses)

        if confusion:
            rvals.append(guesses)
        else:
            correct = tf.math.reduce_sum(tf.linalg.diag_part(guesses))
            rvals.append(correct / total)

        if similarity:
            rvals.append(samples / tf.math.reduce_sum(guesses, axis=1))

        return tuple(rvals)

    """
    Standard call method for static (atemporal) network execution. 
    """
    def call(self, inputs):
        x = self.flatten(inputs)
        if self.projection == "NP":
            x = self.image_encoder(x, training=True)
        elif self.projection == "dot":
            x = tf.multiply(self.direction, x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    """
    Call method for dynamic (temporal) network execution using R&F neurons.
    """
    def call_dynamic(self, inputs, dropout=0.0, jitter=0.0, resolution=-1):
        x = self.flatten(inputs)
        if self.projection == "NP":
            x = self.image_encoder(x, training=True)
        elif self.projection == "dot":
            x = tf.multiply(self.direction, x)
        #convert continuous time representations into periodic spike train
        s = self.phase_to_train(x)
        if dropout > 0.0:
            s = dynamic_dropout(s, dropout)
        if jitter > 0.0:
            s = dynamic_jitter(s, jitter)

        s = self.dense1.call_dynamic(s, dropout=dropout, jitter=jitter, resolution=resolution)
        #don't dropout/jitter at the final layer, do it at input layer instead
        s = self.dense2.call_dynamic(s, dropout=0.0, jitter=0.0, resolution=resolution)
        #convert the spikes back to phases
        y = self.train_to_phase(s, depth=1)

        return np.stack(y, axis=0)

    """
    Given a dataset, produce the network's output phases with either static or dynamic execution.
    """
    def evaluate(self, loader, method="static"):
        outputs = []

        for data in loader:
            x, y = data
            ns = x.shape[0]

            if method == "dynamic":
                yh = self.call_dynamic(x)
                
            else:
                yh = self.call(x)

            outputs.append(yh)

        return tf.concat(outputs, axis=0)

    """
    Given a training set, train the network using standard gradient descent over a number of epochs with static execution.
    """
    def train(self, loader, epochs, report_interval=100):
        losses = []

        for _ in range(epochs):
            for step, data in enumerate(loader):
                x, y = data

                loss = self.train_step(x, y)
                losses.append(loss)

                if step % report_interval == 0:
                    print("Training loss", loss)

        return np.array(losses)

    """
    Convert a list of inputs into a list of spikes representing that input as a delay
    """
    def phase_to_train(self, x):
        n_batch = x.shape[0]
        period = self.dyn_params["period"]
        reps = self.repeats
        
        output = []

        for b in range(n_batch):
            n = x.shape[1]
            t_phase0 = period/2.0

            inds = np.arange(0,n)
            inds = np.tile(inds, reps)

            times = x[b,:] * t_phase0 + t_phase0
            times = np.tile(times, reps)

            #create a list of time offsets to move spikes forward by T for repetitions
            offsets = np.arange(0, reps).reshape(-1,1)
            offsets = np.repeat(offsets, n, axis=1)
            offsets = np.ravel(offsets) * period

            times += offsets
            times = times.ravel()
            
            output.append( (inds, times) )
        
        return output

    """
    Given a series of output phases, produce the predicted class from these outputs. 
    """
    def predict(self, yh, method="static"):
        if method=="static":
            return self._static_prediction(yh)

        elif method=="dynamic_mode":
            return self._mode_prediction(yh)

        else:
            return self._mean_prediction(yh)


    """
    Given a vector of int-based classes, convert these values to a vector of phases.
    """
    def to_phase(self, y):
        onehot = lambda x: be.one_hot(x, self.n_classes)
        return onehot(y) * self.onehot_phase + self.onehot_offset

    """
    Given a batch of example inputs and outputs, carry out a single step of training.
    """
    def train_step(self, x, y):
        y_phase = self.to_phase(y)

        with tf.GradientTape() as tape:
            yh = self.call(x)
            loss = vsa_loss(yh, y_phase)

        trainable_vars = [*self.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss

    """
    Return the series of phases produced at output for each cycle
    """
    def train_to_phase(self, trains, depth=0):
        n_b = len(trains)
        repeats = self.repeats
        period = self.dyn_params["period"]

        outputs = []
        for i in range(n_b):
            out_i, out_t = trains[i]
            phases = np.nan * np.zeros((self.repeats, self.n_classes), "float")

            #assuming eigenfrenquency is same as 1/period
            offset = period/4.0
            #compensate for the delay in turning current into voltage & moving deeper in layers
            offset = (depth+1)*offset

            for j in range(repeats):
                #find the start and stop of the period for this cycle
                tcenter = j*period + offset
                halfcycle = period/2.0
                tstart = tcenter - halfcycle
                tstop =  tcenter + halfcycle
                
                #grab the corresponding indices and values from the solution
                inds = (out_t > tstart) * (out_t < tstop)
                
                for ind in inds:
                    phase = (out_t[ind] - offset) % period
                    phase = 2.0 * phase - 1.0

                    phases[j, out_i[ind]] = phase

            outputs.append(phases)

        return outputs

"""
More modern convolutional architecture which has 4 blocks (input, conv1, conv2, dense) which is used to classify
the CIFAR-10 dataset. 
"""
class Conv2DPhasorModel(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(Conv2DPhasorModel, self).__init__()
        #static parameters
        #number of neurons in the first dense layer in the last block
        self.n_hidden = kwargs.get("n_hidden", 1000)
        #number of classes in the dataset
        self.n_classes = kwargs.get("n_classes", 10)
        #the 'DC' offset for all output neurons phase
        self.onehot_offset = kwargs.get("onehot_offset", 0.0)
        #the offset the active class will be pushed to (quadrature by default)
        self.onehot_phase = kwargs.get("onehot_phase", 0.5)
        #projection used at the front end - random projection not yet implemented here
        self.projection = kwargs.get("projection", "dot")
        #dropout rate used at the end of each block & in the dense block
        self.dropout_rate = kwargs.get("dropout_rate", 0.25)
        #pooling method to use: min, mean, max, or none
        self.pooling = kwargs.get("pooling", "min")
        #L2 regularization to apply to convolutional kernel weights
        self.weight_decay = kwargs.get("weight_decay", 1e-4)

        #dynamic parameters
        self.dyn_params = {
            #how fast potentials in the R&F neuron will decay (negative values cause decay)
            "leakage" : kwargs.get("leakage", -0.2),
            #the eigenperiod for the R&F neuron
            "period" : kwargs.get("period", 1.0),
            #the width of the box current produced by a spike
            "window" : kwargs.get("window", 0.05),
            #spike detection mode ('gradient' or 'cyclemax')
            "spk_mode" : kwargs.get("spk_mode", "gradient"),
            #imaginary (voltage) threshold of the R&F neuron
            "threshold" : kwargs.get("threshold", 0.03),
            #how long the dynamic network will be executed for
            "exec_time" : kwargs.get("exec_time", 10.0),
            #maximum dt for the differential solver
            "max_step" : kwargs.get("max_step", 0.02),
        }
        #how many cycles of spikes will be presented as a dynamic input
        self.repeats = kwargs.get("repeats", 10)

        self.image_shape = input_shape
        self.n_feat = tf.reduce_prod(self.image_shape).numpy().astype('int')

        if self.projection == "dot":
            #define the RPP projection
            self.direction = 2.0 * (tf.cast(tf.random.uniform((1, *self.image_shape)) > 0.5, dtype="float")) - 1.0
            self.project_fn = lambda x: tf.multiply(self.direction, x)
        else:
            #no projection
            self.project_fn = lambda x: x

        self.batchnorm = layers.BatchNormalization()
        self.conv1 = CmpxConv2D(32, (3,3), **self.dyn_params, weight_decay=self.weight_decay, name="conv1")
        self.conv2 = CmpxConv2D(32, (3,3), **self.dyn_params, weight_decay=self.weight_decay, name="conv2")
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout1")
        
        self.conv3 = CmpxConv2D(64, (3,3), **self.dyn_params, weight_decay=self.weight_decay, name="conv3")
        self.conv4 = CmpxConv2D(64, (3,3), **self.dyn_params, weight_decay=self.weight_decay, name="conv4")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout2")

        if self.pooling == "min":
            #do a minpool / WTA by just inverting a maxpool
            self.pool_layer1 = layers.MaxPool2D((2,2))
            self.pool_layer2 = layers.MaxPool2D((2,2))

            self.pool1 = lambda x: -1.0 * self.pool_layer1(-1.0 * x)
            self.pool2 = lambda x: -1.0 * self.pool_layer2(-1.0 * x)

        elif self.pooling == "max":
            self.pool1 = layers.MaxPool2D((2,2))
            self.pool2 = layers.MaxPool2D((2,2))

        elif self.pooling == "mean":
            self.pool1 = layers.AvgPool2D((2,2))
            self.pool2 = layers.AvgPool2D((2,2))

        else:
            self.pool1 = lambda x: x
            self.pool2 = lambda x: x


        self.flatten = layers.Flatten()
        self.dense1 = CmpxLinear(self.n_hidden, **self.dyn_params, name="complex1")
        self.dropout3 = layers.Dropout(self.dropout_rate, name="dropout3")
        self.dense2 = CmpxLinear(self.n_classes, **self.dyn_params, name="complex2") 

    """
    Predict the output class using the last full cycle of phases in a dynamic execution
    """
    def _predict_last(self, phases):
        yh_i = self._predict_ind(phases, ind=-2)
    
        return yh_i

    """
    Predict the output class given a single set of output phases during a single dynamic cycle
    """
    def _predict_ind(self, phases, ind=-2):
        last_phases = phases[:,ind,:]
        dists = np.abs(last_phases - self.onehot_phase)
        yh_i = np.argmin(dists, axis=1)
    
        return yh_i

    """
    Predict the output class for a single vector of phases produced from static execution
    """
    def _static_prediction(self, yh):
        yh_i = tf.argmin(tf.math.abs(yh - self.onehot_phase), axis=1)
        return yh_i


    """
    Measure the accuracy of the network on a test set using either exeuction method.
    Optionally produces a confusion matrix and similarity matrix. 
    """
    def accuracy(self, loader, confusion=True,  similarity=False, method="static"):
        guesses = np.zeros((self.n_classes, self.n_classes), dtype=np.int)

        if similarity:
            samples = np.zeros_like(guesses, dtype=np.float)

        for data in loader:
            x, y = data
            ns = x.shape[0]

            if method == "dynamic_mean" or method == "dynamic":
                yh = self.call_dynamic(x)
                yh_i = self._mean_prediction(yh)

            elif method == "dynamic_mode":
                yh = self.call_dynamic(x)
                yh_i = self._mode_prediction(yh)

            else:
                yh = self.call(x)
                yh_i = self._static_prediction(yh)

            for i in range(ns):
                guesses[y[i], yh_i[i]] += 1

                if similarity:
                    samples[y[i],:] += similarity(yh, self.to_phase(y))

        rvals = []
        total = tf.math.reduce_sum(guesses)

        if confusion:
            rvals.append(guesses)
        else:
            correct = tf.math.reduce_sum(tf.linalg.diag_part(guesses))
            rvals.append(correct / total)

        if similarity:
            rvals.append(samples / tf.math.reduce_sum(guesses, axis=1))

        return tuple(rvals)

    """
    Standard call method for static (atemporal) network execution. 
    """
    def call(self, inputs):
        #input layers (real domain)
        x = self.project_fn(inputs)
        x = self.batchnorm(x)

        #process layers (phasor domain)
        #conv block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        #conv block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        #dense layers & output
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)

        return x

    """
    Call method for dynamic (temporal) network execution using R&F neurons.
    """
    def call_dynamic(self, inputs, dropout=0.0, jitter=0.0, resolution=-1):
        assert self.pooling == "min", "Dynamic execution currently only supports min-pool."

        exec_options = {"dropout": dropout, 
                        "jitter": jitter,
                        "resolution": resolution,
                        }
        x = self.project_fn(inputs)
        x = self.batchnorm(x)
        #convert continuous time representations into periodic spike train
        s = phase_to_train(x, shape=self.image_shape, period=self.dyn_params["period"], repeats=self.repeats)
        if dropout > 0.0:
            s = dynamic_dropout(s, dropout)
        if jitter > 0.0:
            s = dynamic_jitter(s, jitter)

        #conv block 1
        print("Dynamic Execution: Conv 1")
        s = self.conv1.call_dynamic(s, **exec_options)
        #don't dropout before pooling, apply it after
        s = self.conv2.call_dynamic(s, dropout=0.0, jitter=jitter, resolution=resolution)
        s = dynamic_minpool2D(s, self.conv2.output_shape2, self.pool_layer1.pool_size, depth=2)
        if dropout > 0.0:
            s = dynamic_dropout(s, dropout)

        #conv block 2
        print("Dynamic Execution: Conv 2")
        s = self.conv3.call_dynamic(s, **exec_options)
        s = self.conv4.call_dynamic(s, dropout=0.0, jitter=jitter, resolution=resolution)
        s = dynamic_minpool2D(s, self.conv4.output_shape2, self.pool_layer2.pool_size, depth=4)
        if dropout > 0.0:
            s = dynamic_dropout(s, dropout)

        current_shape = self.pool_layer2.compute_output_shape([None, *self.conv4.output_shape2])[1:]
        s = dynamic_flatten(s, current_shape)

        #dense block & output 
        print("Dynamic Execution: Dense")
        s = self.dense1.call_dynamic(s, **exec_options)
        #don't dropout at final layer
        s = self.dense2.call_dynamic(s, dropout=0.0, jitter=0.0, resolution=resolution)
        #convert the spikes back to phases
        y = train_to_phase(s, self.dense2.output_shape2, depth=6, repeats=self.repeats, period=self.dyn_params["period"])

        return np.stack(y, axis=0)

    """
    Given a dataset, produce the network's output phases with either static or dynamic execution.
    """
    def evaluate(self, loader, method="static"):
        outputs = []

        for data in loader:
            x, y = data
            ns = x.shape[0]

            if method == "dynamic":
                yh = self.call_dynamic(x)
                
            else:
                yh = self.call(x)

            outputs.append(yh)

        return tf.concat(outputs, axis=0)

    """
    Given a training set, train the network using standard gradient descent over a number of epochs with static execution.
    """
    def train(self, loader, epochs, report_interval=100):
        losses = []

        for _ in range(epochs):
            for step, data in enumerate(loader):
                x, y = data

                loss = self.train_step(x, y)
                losses.append(loss)

                if step % report_interval == 0:
                    print("Training loss", loss)

        return np.array(losses)
 
    """
    Given a dataflow which can produce an arbitrary number of images to form an augmented dataset, train for a pre-set number of batches.
    """
    def train_aug(self, loader, batches, report_interval=100):
        losses = []

        data_iter = iter(loader)
        for step in range(batches):
            x, y = next(data_iter)

            loss = self.train_step(x, y)
            losses.append(loss)

            if step % report_interval == 0:
                print("Training loss", loss)

        return np.array(losses)
        
    """
    Given a series of output phases, produce the predicted class from these outputs. 
    """
    def predict(self, yh, method="static"):
        if method=="static":
            return self._static_prediction(yh)

        elif method=="dynamic":
            return self._predict_last(yh)

        else:
            return self._mean_prediction(yh)


    """
    Given a vector of int-based classes, convert these values to a vector of phases.
    """
    def to_phase(self, y):
        onehot = lambda x: be.one_hot(x, self.n_classes)
        return onehot(y) * self.onehot_phase + self.onehot_offset

    """
    Given a batch of example inputs and outputs, carry out a single step of training.
    """
    def train_step(self, x, y):
        y_phase = self.to_phase(y)

        with tf.GradientTape() as tape:
            yh = self.call(x)
            loss = vsa_loss(yh, y_phase)

        trainable_vars = [*self.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss