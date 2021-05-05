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

class CmpxLinear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CmpxLinear, self).__init__()
        self.units = units
        #dynamic execution constants
        self.leakage = kwargs.get("leakage", -0.2)
        self.period = kwargs.get("period", 1.0)
        #set the eigenfrequency to 1/T
        self.ang_freq = 2 * np.pi / self.period
        self.window = kwargs.get("window", 0.05)
        self.spk_mode = kwargs.get("spk_mode", "gradient")
        self.threshold = kwargs.get("threshold", 0.03)
        self.exec_time = kwargs.get("exec_time", 10.0)
        self.max_step = kwargs.get("max_step", 0.01)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w"
        )
        self.n_in = self.w.shape[0]

        #add bias?
        # self.b = self.add_weight(
        #     shape=self.units,
        #     initializer="zeros",
        #     trainable=True
        # )

    def call(self, inputs, mode="static"):
        if mode=="dynamic":
            output = self.call_dynamic(inputs)
        else:
            output = self.call_static(inputs)

        return output

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

    def dz(self, t, z, current):
        k = tf.complex(self.leakage, self.ang_freq)
        
        #scale currents by synaptic weight
        currents = tf.matmul(current(t), self.w)
        currents = tf.complex(currents, tf.zeros_like(currents))
        
        dz = k * z + currents
        return dz.numpy()

    #currently inference only
    def call_dynamic(self, inputs):
        solutions = []
        outputs = []
        n_batches = len(inputs)

        for i in tqdm(range(n_batches)):
            input_i = inputs[i][0]
            input_t = inputs[i][1]
            z0 = np.zeros((self.units), "complex")

            i_fn = lambda t: self.current(t, (input_i, input_t))
            dz_fn = lambda t,z: self.dz(t, z, i_fn)
            sol = solve_ivp(dz_fn, (0.0, self.exec_time), z0, max_step=self.max_step)
            solutions.append(sol)

            if self.spk_mode == "gradient":
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            elif self.spk_mode == "cyclemax":
                spk = findspks_max(sol, threshold=self.threshold, period=self.period)
            else:
                print("WARNING: Spike mode not recognized, defaulting to gradient")
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            
            spk_inds, spk_tms = np.nonzero(spk)
            spk_tms = sol.t[spk_tms]

            outputs.append( (spk_inds, spk_tms) )

        self.solutions = solutions
        self.spike_trains = outputs
        return outputs


    def call_static(self, inputs):
        pi = tf.constant(np.pi)
        #clip inputs to -1, 1 domain (pi-normalized phasor)
        inputs = tf.clip_by_value(inputs, -1, 1)
        #convert the phase angles into complex vectors
        inputs = phase_to_complex(inputs)
        #scale the complex vectors by weight and sum
        inputs = tf.matmul(inputs, tf.complex(self.w, tf.zeros_like(self.w)))
        #convert them back to phase angles
        output = tf.math.angle(inputs) / pi

        self.input_shape2 = inputs.shape[1:]
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

class CmpxConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CmpxConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.weight_decay = kwargs.get("weight_decay", 1e-4)

        #create the convolutional operation via a sub-layer
        self.operation = layers.Conv2D(self.filters, 
                        kernel_size=self.kernel_size,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(self.weight_decay))
        
        #dynamic execution constants
        self.leakage = kwargs.get("leakage", -0.2)
        self.period = kwargs.get("period", 1.0)
        #set the eigenfrequency to 1/T
        self.ang_freq = 2 * np.pi / self.period
        self.window = kwargs.get("window", 0.05)
        self.spk_mode = kwargs.get("spk_mode", "gradient")
        self.threshold = kwargs.get("threshold", 0.03)
        self.exec_time = kwargs.get("exec_time", 10.0)
        self.max_step = kwargs.get("max_step", 0.01)

    def build(self, input_shape):
        self.operation.build(input_shape)

        
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

    #currently inference only
    def call_dynamic(self, inputs, save_solutions=False):
        solutions = []
        outputs = []
        n_batches = len(inputs)

        out_shape = self.output_shape2
        n_neurons = np.prod(out_shape)

        for i in tqdm(range(n_batches)):
            input_i = inputs[i][0]
            input_t = inputs[i][1]
            
            z0 = np.zeros(n_neurons, "complex")

            i_fn = lambda t: self.current(t, (input_i, input_t))
            dz_fn = lambda t,z: self.dz(t, z, i_fn)
            sol = solve_ivp(dz_fn, (0.0, self.exec_time), z0, max_step=self.max_step)
            if save_solutions:
                solutions.append(sol)

            if self.spk_mode == "gradient":
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            elif self.spk_mode == "cyclemax":
                spk = findspks_max(sol, threshold=self.threshold, period=self.period)
            else:
                print("WARNING: Spike mode not recognized, defaulting to gradient")
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            
            spks = tf.where(spk > 0.0)
            spk_tms = tf.gather(sol.t, spks[:,1])
            spk_tms = tf.cast(spk_tms, "float")
            spk_inds = tf.unravel_index(spks[:,0], dims=out_shape)
            
            outputs.append( (spk_inds, spk_tms) )

        self.solutions = solutions
        self.spike_trains = outputs
        return outputs


    def call_static(self, inputs):
        return call(input)

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

    def dz(self, t, z, current):
        #constant which defines leakage, oscillation
        k = tf.complex(self.leakage, self.ang_freq)
        
        real_output = self.operation(current(t))
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