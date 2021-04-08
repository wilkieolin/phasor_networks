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
Converts a vector of features from a detector (e.g. convolutional net, ResNet) and projects/
normalizes them into a VSA symbol.
"""
class VSA_Encoder(keras.Model):
    def __init__(self, n_in, n_out, overscan=1.0, sigma=3.0, name='VSA Encoder'):
        super(VSA_Encoder, self).__init__()
        #self.norm_features = SBatchNorm(n_in, 1.0)
        self.transform = StaticLinear(n_in, n_out, overscan)
        self.norm_symbols = Normalize(sigma)

        # self.inv_transform = keras.Sequential([
        #     keras.layers.Input(n_out),
        #     keras.layers.Dropout(0.5),
        #     keras.layers.Dense(n_in)
        # ])
        
    def call(self, inputs, **kwargs):
        #x = self.norm_features(inputs, **kwargs)
        x = self.transform(inputs)
        x = self.norm_symbols(x, **kwargs)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x

    def reverse(self, inputs):
        x = self.norm_symbols.reverse(inputs)
        x = self.transform.reverse(x)
        #x = self.norm_features.reverse(x)
        

        return x    

class PhasorModel(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(PhasorModel, self).__init__()
        #static parameters
        self.overscan = kwargs.get("overscan", 100)
        self.sigma = kwargs.get("sigma", 3.0)
        self.n_hidden = kwargs.get("n_hidden", 100)
        self.n_classes = kwargs.get("n_classes", 10)
        self.onehot_offset = kwargs.get("onehot_offset", -0.5)
        self.onehot_phase = kwargs.get("onehot_phase", 1.0)
        self.projection = kwargs.get("projection", "dot")
        self.dropout_rate = kwargs.get("dropout_rate", 0.25)

        #dynamic parameters
        self.dyn_params = {
            "leakage" : kwargs.get("leakage", -0.2),
            "period" : kwargs.get("period", 1.0),
            "window" : kwargs.get("window", 0.05),
            "spk_mode" : kwargs.get("spk_mode", "gradient"),
            "threshold" : kwargs.get("threshold", 0.03),
            "exec_time" : kwargs.get("exec_time", 10.0),
            "max_step" : kwargs.get("max_step", 0.02)
        }
        self.repeats = kwargs.get("repeats", 10)

        n_feat = tf.reduce_prod(input_shape).numpy().astype('int')

        self.flatten = layers.Flatten(name="flatten")
        if self.projection == "NP":
            self.image_encoder = VSA_Encoder(n_feat, n_feat, overscan=self.overscan, sigma=self.sigma)
        elif self.projection == "dot":
            self.direction = 2.0 * (tf.cast(tf.random.uniform((1, n_feat)) > 0.5, dtype="float")) - 1.0
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout1")
        self.dense1 = CmpxLinear(self.n_hidden, **self.dyn_params, name="complex1")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout2")
        self.dense2 = CmpxLinear(self.n_classes, **self.dyn_params, name="complex2")

    def _mean_prediction(self, yh):
        #take the average across phases
        yh_avg = np.nanmean(yh, axis=1)
        yh_i = np.argmin(np.abs(yh_avg - self.onehot_phase), axis=1)
        return yh_i

    def _mode_prediction(self, yh):
        yh_i = np.argmin(np.abs(yh - self.onehot_phase), axis=2)
        yh_i = mode(yh_i, axis=1)[0]
        return yh_i.ravel()

    def _static_prediction(self, yh):
        yh_i = tf.argmin(tf.math.abs(yh - self.onehot_phase), axis=1)
        return yh_i


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

    def call_dynamic(self, inputs):
        x = self.flatten(inputs)
        if self.projection == "NP":
            x = self.image_encoder(x, training=True)
        elif self.projection == "dot":
            x = tf.multiply(self.direction, x)
        #convert continuous time representations into periodic spike train
        s = self.phase_to_train(x)
        s = self.dense1.call_dynamic(s)
        s = self.dense2.call_dynamic(s)
        #convert the spikes back to phases
        y = self.train_to_phase(s, depth=1)

        return np.stack(y, axis=0)

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

    def predict(self, yh, method="static"):
        if method=="static":
            return self._static_prediction(yh)

        elif method=="dynamic_mode":
            return self._mode_prediction(yh)

        else:
            return self._mean_prediction(yh)


    def to_phase(self, y):
        onehot = lambda x: be.one_hot(x, self.n_classes)
        return onehot(y) * self.onehot_phase + self.onehot_offset

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
                #print(str(tstart) + " " + str(tstop))
                
                
                #grab the corresponding indices and values from the solution
                inds = (out_t > tstart) * (out_t < tstop)
                
                for ind in inds:
                    phase = (out_t[ind] - offset) % period
                    phase = 2.0 * phase - 1.0

                    phases[j, out_i[ind]] = phase

            outputs.append(phases)

        return outputs


        

        