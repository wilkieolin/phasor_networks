"""
This file defines the layers which are composed to form deep phasor networks.
The methods to execute these layers either with respect to time (temporal/dynamic)
or without regards to it (atemporal/static) are defined for each layer.

Wilkie Olin-Ammentorp, 2021
University of Califonia, San Diego
"""

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers

from utils import *
from scipy.integrate import solve_ivp
from tqdm import tqdm

"""
A dense layer which uses sparse weights to do a random projection which is not trainable.
"""
class StaticLinear(keras.layers.Layer):
    def __init__(self, n_in, n_out, overscan=1.0):
        super(StaticLinear, self).__init__()
        
        self.w = tf.Variable(
            initial_value = construct_sparse(n_in, n_out, overscan),
            trainable = False)

        self.rev_w = tf.linalg.pinv(self.w)

    def reverse(self, inputs):
        return tf.matmul(inputs, self.rev_w)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w)

"""
Layer which defines the operations needed to carry out a standard dense layer of a neural network
using the temporal/atemporal phasor methods.
"""
class CmpxLinear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CmpxLinear, self).__init__()
        #number of units in the layer
        self.units = units
        #dynamic execution constants - these are generally passed down from the model when the layer is defined.
        self.leakage = kwargs.get("leakage", -0.2)
        self.period = kwargs.get("period", 1.0)
        #set the eigenfrequency to 1/T
        self.ang_freq = 2 * np.pi / self.period
        self.window = kwargs.get("window", 0.05)
        self.spk_mode = kwargs.get("spk_mode", "gradient")
        self.threshold = kwargs.get("threshold", 0.03)
        self.exec_time = kwargs.get("exec_time", 10.0)
        self.max_step = kwargs.get("max_step", 0.01)

    """
    Add the weights and calculate other parameters needed for execution after construction.
    """
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w"
        )
        self.n_in = self.w.shape[0]

    """
    Calculate the output phases given an input layer and execution method.
    """
    def call(self, inputs, mode="static"):
        if mode=="dynamic":
            output = self.call_dynamic(inputs)
        else:
            output = self.call_static(inputs)

        return output

    """
    Given the current time in the calculation and the input spike train, calculate 
    the currents which are being supplied to each R&F neuron's input synapses.
    """
    def current(self, t, spikes):
        spikes_i, spikes_t = spikes
        window = self.window
        
        shape = self.input_shape2
        cast = lambda x: tf.cast(x, "float")
        box_start = cast(tf.constant(t - window))
        box_end = cast(tf.constant(t + window))

        #determine which spikes at this time are active
        pre = cast(spikes_t < box_end)
        post = cast(spikes_t > box_start)
        active_i = tf.where(pre * post)[:,0]

        #get the indices of the neurons those active times correspond to
        active_i = tf.gather(spikes_i, active_i, axis=0)
        active_i = tf.cast(active_i, "int64")
        
        n_active = active_i.shape[0]
        
        if n_active > 0:
            active_i = tf.expand_dims(active_i, 1)
            updates = tf.ones(n_active)
            currents = tf.scatter_nd(active_i, updates, shape=shape)
        else:
            currents = tf.zeros(shape, dtype="float")
        
        #make it 2D for the matrix multiply
        return tf.expand_dims(currents, 0)

    """
    Calculate the differential changes in R&F neuron potential for an instant in time given a time, previous potentials,
    and a lambda to produce currents given a time (parent function defined above). 
    """
    def dz(self, t, z, current):
        #the leakage and oscillation parameters are combined to a single complex constant, k
        k = tf.complex(self.leakage, self.ang_freq)
        
        #scale calculated currents by the input synaptic weights
        currents = tf.matmul(current(t), self.w)
        #convert these real values to the complex domain
        currents = tf.complex(currents, tf.zeros_like(currents))
        
        #update the previous potential and add the currents
        dz = k * z + currents
        return dz.numpy()

    """
    Given a series of input spike trains (a list with a tuple of (indices, times) for each example), 
    carry out the dynamic/temporal execution for these inputs and return a spike train. Optionally, save
    the full solutions through time of the complex potentials (memory-hungry op).

    Training cannot be done currently through this op as it calls numpy/scipy differential solvers & not an adjoint-based one.
    """
    def call_dynamic(self, inputs, dropout=0.0, jitter=0.0, save_solutions=False):
        #array to save full solutions in
        solutions = []
        #array to save the output spike trains in
        outputs = []
        #number of examples in the input
        n_batches = len(inputs)

        for i in tqdm(range(n_batches)):
            #extract the spike indices and times for this input
            input_i = inputs[i][0]
            input_t = inputs[i][1]
            #initialize the complex potentials of the R&F neuron
            z0 = np.zeros((self.units), "complex")

            #define the lambda function to produce currents at any time given this spike train
            i_fn = lambda t: self.current(t, (input_i, input_t))
            #define the lambda function which updates potentials through time
            dz_fn = lambda t,z: self.dz(t, z, i_fn)
            #call scipy differential solver
            sol = solve_ivp(dz_fn, (0.0, self.exec_time), z0, max_step=self.max_step)
            if save_solutions:
                #save the full solutions if desired
                solutions.append(sol)

            #detect spikes from the output potentials
            if self.spk_mode == "gradient":
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            elif self.spk_mode == "cyclemax":
                spk = findspks_max(sol, threshold=self.threshold, period=self.period)
            else:
                print("WARNING: Spike mode not recognized, defaulting to gradient")
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            
            #convert the dense boolean matrix of detect spikes to a sparse spike train
            spk_inds, spk_tms = np.nonzero(spk)
            spk_tms = sol.t[spk_tms]

            #append the solution's sparse spike train to the outputs
            outputs.append( (spk_inds, spk_tms) )

        if dropout > 0.0:
            self.outputs = dynamic_dropout(outputs, dropout)

        if jitter > 0.0:
            self.outputs = dynamic_jitter(outputs, jitter)

        self.solutions = solutions
        self.spike_trains = outputs
        return outputs

    """
    Given a set of input phases, calculate the output phases using the atemporal/static method.
    """
    def call_static(self, inputs):
        #provide a second internal reference to input/output shapes on calling to avoid some awkwardness in current keras ops
        self.input_shape2 = inputs.shape[1:]

        pi = tf.constant(np.pi)
        #clip inputs to -1, 1 domain (pi-normalized phasor)
        x = tf.clip_by_value(inputs, -1, 1)
        #convert the phase angles into complex vectors
        x = phase_to_complex(x)
        #scale the complex vectors by weight and sum
        x = tf.matmul(x, tf.complex(self.w, tf.zeros_like(self.w)))
        #convert them back to phase angles
        output = tf.math.angle(x) / pi

        self.output_shape2 = output.shape[1:]

        return output


    def get_config(self):
        config = super(CmpxLinear, self).get_config()
        config.update({"units": self.units})
        config.update({"w": self.w.numpy()})
        return config

    def get_weights(self):
        return [self.w.numpy()]

    def set_weights(self, weights):
        self.w.value = weights[0]

"""
Convolutional layer which operates using phasor activations.
"""
class CmpxConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CmpxConv2D, self).__init__()
        #standard 2D convolutional parameters: number of filters and kernel size
        self.filters = filters
        self.kernel_size = kernel_size
        #weight decay used to L2 regularize kernels
        self.weight_decay = kwargs.get("weight_decay", 1e-4)

        #implement the convolutional operation via a standard keras sub-layer
        self.operation = layers.Conv2D(self.filters, 
                        kernel_size=self.kernel_size,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(self.weight_decay))
        
        #dynamic execution constants - inherited from model
        self.leakage = kwargs.get("leakage", -0.2)
        self.period = kwargs.get("period", 1.0)
        #set the eigenfrequency to 1/T
        self.ang_freq = 2 * np.pi / self.period
        self.window = kwargs.get("window", 0.05)
        self.spk_mode = kwargs.get("spk_mode", "gradient")
        self.threshold = kwargs.get("threshold", 0.03)
        self.exec_time = kwargs.get("exec_time", 10.0)
        self.max_step = kwargs.get("max_step", 0.01)

    """
    Method which builds the convolutional operation after construction.
    """
    def build(self, input_shape):
        self.operation.build(input_shape)

    """
    Method to calculate static output phases from a series of input images/channels. 
    """
    def call(self, inputs):
        pi = tf.constant(np.pi)
        
        #convert the phase angles into complex vectors
        inputs = phase_to_complex(inputs)
        #scale the complex vectors by weight and sum
        real_output = self.operation(tf.math.real(inputs))
        imag_output = self.operation(tf.math.imag(inputs))
        output = tf.complex(real_output, imag_output)
        #convert them back to phase angles
        output = tf.math.angle(output) / pi

        #this isn't computed automatically via call for some reason
        self.input_shape2 = inputs.shape[1:]
        self.output_shape2 = output.shape[1:]

        return output

    """
    Given a series of input spike trains (a list with a tuple of (3-D indices, times) for each example), 
    carry out the dynamic/temporal execution for these inputs and return a spike train. Optionally, save
    the full solutions through time of the complex potentials (memory-hungry op).

    Training cannot be done currently through this op as it calls numpy/scipy differential solvers & not an adjoint-based one.
    """
    def call_dynamic(self, inputs, dropout=0.0, jitter=0.0, save_solutions=False):
        #array to save full solutions in
        solutions = []
        #array to save the output spike trains in
        outputs = []
        #number of examples in the input
        n_batches = len(inputs)

        #calculated output shape & number of features
        out_shape = self.output_shape2
        n_features = np.prod(out_shape)

        for i in tqdm(range(n_batches)):
            #extract the spike indices and times for this input
            input_i = inputs[i][0]
            input_t = inputs[i][1]
            #initialize the complex potentials of the R&F neuron
            z0 = np.zeros(n_features, "complex")

            #define the lambda function to produce currents at any time given this spike train
            i_fn = lambda t: self.current(t, (input_i, input_t))
            #define the lambda function which updates potentials through time
            dz_fn = lambda t,z: self.dz(t, z, i_fn)
            #call scipy differential solver
            sol = solve_ivp(dz_fn, (0.0, self.exec_time), z0, max_step=self.max_step)
            if save_solutions:
                solutions.append(sol)

            #detect spikes from the output potentials
            if self.spk_mode == "gradient":
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            elif self.spk_mode == "cyclemax":
                spk = findspks_max(sol, threshold=self.threshold, period=self.period)
            else:
                print("WARNING: Spike mode not recognized, defaulting to gradient")
                spk = findspks(sol, threshold=self.threshold, period=self.period)
                        
            #find the vectoral (1-D) indices & times where spikes happened
            spks = tf.where(spk)
            spk_tms = tf.gather(sol.t, spks[:,1])
            spk_tms = tf.cast(spk_tms, "float")
            #convert the vectoral indices back to 3D coordinates
            spk_inds = tf.unravel_index(spks[:,0], dims=out_shape)
            
            outputs.append( (spk_inds, spk_tms) )

        if dropout > 0.0:
            self.outputs = dynamic_dropout(outputs, dropout)

        if jitter > 0.0:
            self.outputs = dynamic_jitter(outputs, jitter)

        self.solutions = solutions
        self.spike_trains = outputs
        return outputs

    """
    Given the current time in the calculation and the input spike train, calculate 
    the currents which are being supplied to each R&F neuron's input synapses.
    """
    def current(self, t, spikes):
        spikes_i, spikes_t = spikes
        window = self.window
        
        shape = self.input_shape2
        cast = lambda x: tf.cast(x, "float")
        box_start = cast(tf.constant(t - window))
        box_end = cast(tf.constant(t + window))

        #determine which spikes at this time are active
        pre = cast(spikes_t < box_end)
        post = cast(spikes_t > box_start)
        active_i = tf.where(pre * post)[:,0]

        #get the indices of the neurons those active times correspond to
        active_i = tf.gather(spikes_i, active_i, axis=1)
        active_i = tf.cast(active_i, "int64")
        
        n_active = active_i.shape[1]
        
        if n_active > 0:
            active_i = tf.transpose(active_i)
            updates = tf.ones(n_active)
            currents = tf.scatter_nd(active_i, updates, shape=shape)
        else:
            currents = tf.zeros(shape, dtype="float")
        
        #add a 'dummy' batch of 1 for the convolutional input
        currents = tf.expand_dims(currents, axis=0)
        return currents

    """
    Calculate the differential changes in R&F neuron potential for an instant in time given a time, previous potentials,
    and a lambda to produce currents given a time (parent function defined above). 
    """
    def dz(self, t, z, current):
        #constant which defines leakage, oscillation
        k = tf.complex(self.leakage, self.ang_freq)
        
        #carry out the convolution over the input currents
        real_output = self.operation(current(t))
        #currents are defined as real values, im component is zero
        imag_output = tf.zeros_like(real_output)
        currents = tf.complex(real_output, imag_output)

        flatten = lambda x: tf.reshape(x, -1)
        
        dz = k * z + flatten(currents)
        return dz.numpy()

    def get_config(self):
        config = super(CmpxConv2D, self).get_config()
        config.update({"filters", self.filters})
        config.update({"kernel_size", self.kernel_size})
        return config

    def get_weights(self):
        return [self.operation.kernel.numpy()]

    def set_weights(self, weights):
        self.operation.kernel.value = weights[0]
        
"""
Layer which normalizes inputs with 2 self-learned parameters on an input stream.
Similar to batch norm but uses single mean and std. dev across entire input vector.
"""
class Normalize(keras.layers.Layer):
    def __init__(self, sigma, **kwargs):
        super(Normalize, self).__init__()

        self.sigma = tf.constant(sigma)

        momentum = kwargs.get("momentum", 0.99)
        self.momentum = tf.constant(momentum)
        
        epsilon = kwargs.get("epsilon", 0.001)
        self.epsilon = tf.constant(
            epsilon * tf.ones((1,), dtype="float32")
        )

        self.moving_mean = tf.Variable(
            initial_value = tf.zeros((1,), dtype="float32"),
            trainable = False
        )

        self.moving_std = tf.Variable(
            initial_value = tf.ones((1,), dtype="float32"),
            trainable = False
        )

    def call(self, data, **kwargs):
        training = kwargs.get("training", True)

        if training:
            #calculate batch moments
            mean, var = tf.stop_gradient(tf.nn.moments(tf.reshape(data, -1),axes=0))
            std = tf.math.sqrt(var)

            #updating the moving moments
            self.moving_mean = self.moving_mean * self.momentum + mean * (1-self.momentum)
            self.moving_std = self.moving_std * self.momentum + std * (1-self.momentum)

            #batchnorm scaling
            output = (data - mean) / (std + self.epsilon) 
            return output / self.sigma

        else:
            #scale with calculated moments
            mean = self.moving_mean
            std = self.moving_std

            #batchnorm scaling
            output = (data - mean) / (std + self.epsilon)
            return output / self.sigma

    def reverse(self, data):
        #undo scaling with calculated moments
        mean = self.moving_mean
        std = self.moving_std

        x = self.sigma * data
        output = (x)*(std + self.epsilon) + mean
        
        return output