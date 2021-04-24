import tensorflow as tf
import numpy as np

def avgphase(trains, n_n):
    n_b = len(trains)

    outputs = []
    for i in range(n_b):
        nz_i, nz_t = trains[i]

        avgphase = np.zeros(n_n, "float")
        for n in range(n_n):
            inds = nz_i == n
            avgphase[n] = np.mean(nz_t[inds])

        outputs.append(avgphase)
        
    return np.array(outputs)
    
def construct_sparse(n1, n2, overscan=1.0):
    shp = (n1,n2)
    m = np.zeros(shp).ravel()
    
    #the number of times we'll loop through
    while overscan >= 1.0:
        #cutoff by remaining overscan
        cutoff = int(overscan * n2)
        overscan -= 1.0
        
        #the permutation from dest -> source we're assigning this time
        sources = np.floor(n1 * np.random.rand(n2)).astype(np.int)[0:cutoff]
        dests = np.arange(0, n2, 1).astype(np.int)[0:cutoff]
        #shuffle mutates original array
        np.random.shuffle(dests)
        indices = [sources, dests]
        indices = np.ravel_multi_index(indices, shp)
        
        #return indices
        m[indices] = np.random.uniform(low=-1, high=1, size=(len(indices)))
        
    m = np.reshape(m, shp)
            
    return tf.constant(m, dtype="float32")

def dynamic_dropout(trains, rate):
    def dropout_lambda(x):
        indices, times = x
        if len(indices.shape) == 1:
            #for flattened index vectors
            i_gather_axis = 0
        else:
            #for cartesian index matrices
            i_gather_axis = 1
            
        n_x = times.shape[0]
        
        rand_indices = tf.where(tf.random.uniform((n_x,)) > rate)
        d_indices = tf.gather(indices, rand_indices, axis=i_gather_axis)
        d_indices = tf.squeeze(d_indices)
        
        d_times = tf.gather(times, rand_indices, axis=0)
        d_times = tf.squeeze(d_times)
        
        return (d_indices, d_times)
    
    return list(map(dropout_lambda, trains))

def dynamic_flatten(trains, input_shape):
    
    def flatten_lambda(x):
        strides = tf.math.cumprod(input_shape, exclusive=True, reverse=True)
        strides = tf.expand_dims(strides, 0)
        strides = tf.cast(strides, dtype="int64")

        indices = x[0]

        flat_indices = tf.matmul(strides, indices)
        flat_indices = tf.reshape(flat_indices, -1)
        
        return (indices, x[1])

    return list(map(flatten_lambda, trains))

def dynamic_minpool2D(trains, input_shape, pool_size, method="relative", **kwargs):
    assert len(pool_size) == 2, "Must have 2-D pool"
    if method == "relative":
        r_period = kwargs.get("r_period", 0.9)
        r_lambda = lambda x: refract(x, r_period)
    else:
        period = kwargs.get("period", 1.0)
        depth = kwargs.get("depth", 1)
        offset = period / 4.0 + depth*period
        r_lambda = lambda x: refract_absolute(x, period, offset)
    
    index_groups, output_shape = generate_index_groups(input_shape, pool_size)
                            
    #transform an individual spike train by pooling times locally to get the first-firing
    def train_lambda(train):
        
        pool_lambda = lambda x: pool(x, train, r_lambda)
        
        #collect the min-pooled firing times over all pools
        output = list(map(pool_lambda, index_groups))
        
        output_indices = tf.concat([o[0] for o in output], axis=0)
        output_times = tf.concat([o[1] for o in output], axis=0)
        
        return (output_indices, output_times)
         
    return list(map(train_lambda, trains))

def dynamic_unflatten(trains, input_shape):
    unflatten_lambda = lambda x: tf.unravel_index(x[0], input_shape)

    return list(map(unflatten_lambda, trains))


def findspks(sol, threshold=2e-3, refractory=0.25, period=1.0):
    refrac_t = period*refractory
    ts = sol.t
    tmax = ts[-1]
    zs = sol.y
    n_t = sol.t.shape[0]
    n_n = sol.y.shape[0]
    
    #find where voltage reaches its max
    voltage = np.imag(zs)
    dvs = np.gradient(voltage, axis=1)
    dsign = np.sign(dvs)
    spks = np.diff(dsign, axis=1, prepend=np.zeros_like((zs.shape[1]))) < 0
    
    #filter by threshold
    above_t = voltage > threshold
    spks = spks * above_t
    
    #apply the refractory period
    if refractory > 0.0:
        for t in range(n_t):
            #current time + refractory window
            stop = ts[t] + refrac_t
            if stop > tmax:
                stop_i = -1
            else:
                #find the first index where t > stop
                stop_i = np.nonzero(ts > stop)[0][0]
            
            for n in range(n_n):
                #if there's a spike
                if spks[n,t] == True:
                    spks[n,t+1:stop_i] = False

    return spks

def findspks_max(sol, threshold=0.05, period=1.0):
    all_spks = []
    
    #slice the solution into its periods
    zslices = split_by_period(sol, dtype="v", period=period)
    n_periods = len(zslices)
    n_t = sol.t.shape[0]
    n_neurons = sol.y.shape[0]
    
    for i in range(n_periods):
        zslice = zslices[i]
        spk_slice = np.zeros_like(zslice, dtype="float")
        
        vs = np.imag(zslice)
        #find the ind of the maximum voltage value
        make2d = lambda x: x.reshape((n_neurons,1))
        i_maxes = np.argmax(vs, axis=1).reshape((n_neurons,1))
        #return spk_slice, i_maxes
        np.put_along_axis(spk_slice, indices=i_maxes, values=1.0, axis=1)
        
        #take the corresponding value
        max_values = np.max(vs, axis=1)
        positive = max_values > threshold
        positive = make2d(positive)
        #only mark spikes where voltage goes positive
        np.multiply(positive, spk_slice)
        all_spks.append(spk_slice)
        
    all_spks = np.concatenate(all_spks, axis=1)
    missing_t = n_t - all_spks.shape[1]
    all_spks = np.concatenate((all_spks, np.zeros((n_neurons, missing_t))), axis=1)
    return all_spks

"""
Given the inputs, generate the input -> output mapping
"""
def generate_index_groups(input_shape, pool_size):
    n_x, n_y, n_c = input_shape
    k_x, k_y = pool_size
    
    strides_x = n_x // k_x
    strides_y = n_y // k_y
    
    output_shape = (strides_x, strides_y, n_c)
    
    index_groups = []
    output_indices = []
    for c in range(n_c):
        for x in range(strides_x):
            for y in range(strides_y):
                #generate the members bounded by each valid pool
                start_x = k_x * x
                stop_x = start_x + k_x
                x_inds = np.arange(start_x, stop_x)
                
                start_y = k_y * y
                stop_y = start_y + k_y
                y_inds = np.arange(start_y, stop_y)
                
                group = []
                for i_x in x_inds:
                    for i_y in y_inds:
                        coordinate = np.array([i_x, i_y, c])
                        group.append(coordinate)
                        
                #the group of input indices which are pooled to produce the single output index
                index_groups.append(np.array(group))
                output_indices.append(np.array(((x,y,c),)))
                
    kernel_pairs = list(zip(output_indices, index_groups))
                
    return kernel_pairs, output_shape

def limit_gpus():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


"""
Use Euler's identity to quickly convert a vector of phases to complex numbers
for addition / other ops.
"""
#@tf.function
def phase_to_complex(x):
    x = tf.complex(x, tf.zeros_like(x))
    pi = tf.complex(np.pi, 0.0)
    im = tf.complex(0.0, 1.0)
    return tf.exp(im * x * pi)

def phase_to_train(x, shape, period=1.0, repeats=3):
        n_batch = x.shape[0]
        features = np.prod(shape)

        output = []

        for b in range(n_batch):
            t_phase0 = period/2.0

            #create the list of indices with arbitrary dimension
            inds = tf.range(0, features, delta=1)
            inds = tf.tile(inds, [repeats])
            inds = tf.unravel_index(inds, shape)
            
            #list the time offset for each index and repeat it for repeats cycles
            times = x[b,...] * t_phase0 + t_phase0
            times = tf.reshape(times, (-1))
            times = tf.tile(times, [repeats])
            
            #create a list of time offsets to move spikes forward by T for repetitions
            offsets = tf.range(0, repeats, delta=1, dtype="float")
            offsets = tf.repeat(offsets, features, axis=0)
            offsets = tf.reshape(offsets, (-1)) * tf.constant(period)
            
            times += offsets
            
            output.append( (inds, times) )
        
        return output

#given the indices of a single pool, get the minpool values
def pool(pool_mapping, train, refraction):
    #the indices which the current kernel is collecting from
    kernel_output_ind, kernel_input_inds = pool_mapping
    #times and indices from the entire layer
    train_inds, train_times = train

    #lambda which return where all indices match the input
    get_matches = lambda x: tf.reduce_all(train_inds == tf.expand_dims(x, 0), axis=1)
    #map the lambda over all the kernel indices (same as map_fn but works)
    matches = tf.stack([get_matches(ind) for ind in tf.unstack(kernel_input_inds)])
    #then reduce_sum to get all matches & their location
    all_matches = tf.where(tf.reduce_sum(tf.cast(matches, dtype="int32"), axis=0))

    if len(all_matches > 0):
        pool_times = tf.squeeze(tf.gather(train_times, all_matches))

        times = refraction(pool_times)

        n_t = times.shape[0]
        #broadcast the output index to the number of spikes
        indices = tf.tile(kernel_output_ind, (n_t, 1))
    else:
        empty = lambda: tf.constant([])
        indices = empty()
        times = empty()

    return (indices, times)

def refract(times, r_period):
    n_s = times.shape[0]
    #use numpy because TF can be *very* slow at iterated scalar ops
    times = tf.sort(times, axis=0).numpy()
    
    inds = []
    i = 0
    while i < n_s:
        #add the index of the first spike after the refractory period to the list
        inds.append(i)
        t_stop = times[i] + r_period
        
        #find the next non-refractory spike
        while i < n_s and times[i] < t_stop:
            i += 1
            
    refractory_times = tf.gather(times, inds)
    return refractory_times

def refract_absolute(times, period, offset):
    n_s = times.shape[0]
    
    times = tf.sort(times, axis=0).numpy()
    
    inds = []
    i = 0
    p = 1
    
    while i < n_s:
        #add the index of the first spike after the refractory period to the list
        inds.append(i)
        t_stop = period*p + offset
        
        #find the next non-refractory spike
        while i < n_s and times[i] < t_stop:
            i += 1
            
        p += 1
            
    refractory_times = tf.gather(times, inds)
    return refractory_times

"""
Move phases from (-inf, inf) into (-pi, pi)
"""
#@tf.function
def remap_phase(x):
    #move from (-inf, inf) to (0,2)
    n1 = tf.constant(-1.0)
    tau = tf.constant(2.0)
    pi = tf.constant(1.0)
    
    x = tf.math.floormod(x, tau)
    #move (1,2) to (-1, 0) to be consistent with tanh activations
    return n1 * tau * tf.cast(tf.math.greater(x, pi), tf.float32) + x


def split_by_period(sol, dtype="v", period=1.0):
    if dtype=="v":
        #looking at the voltage
        offset = period*0.25
    else:
        #looking at the current
        offset = 0.0
        
    ts = sol.t
    zs = sol.y
    tmax = ts[-1]
    periods = int(tmax // period)
    
    slices = []
    for i in range(periods):
        #find the start and stop of the period for this cycle
        tcenter = i*period + offset
        halfcycle = period/2.0
        tstart = tcenter - halfcycle
        tstop =  tcenter + halfcycle
        #print(str(tstart) + " " + str(tstop))
        
        #grab the corresponding indices and values from the solution
        inds = (ts > tstart) * (ts < tstop)
        zslice = zs[:,inds]
        slices.append(zslice)
        
    return slices
    
"""
Restrict visible GPUs since TF is a little greedy
"""
def set_gpu(idx):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[idx], 'GPU')

"""
Return the similarity of two phase vectors defined by the FHNN framework
"""
#@tf.function
def similarity(x, y):
    assert x.shape == y.shape, "Function is for comparing similarity of tensors with identical shapes and 1:1 mapping: " + str(x.shape) + " " + str(y.shape)
    pi = tf.constant(np.pi)
    return tf.math.reduce_mean(tf.math.cos(pi*(x - y)), axis=1)

def train_to_phase(trains, shape, depth=0, repeats=3, period=1.0):
    n_b = len(trains)

    outputs = []
    for i in range(n_b):
        out_i, out_t = trains[i]
        all_phases = np.nan * np.zeros((repeats, *shape), "float")
        #all_phases = []
        
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
            cast = lambda x: tf.cast(x, "float")
            
            inds = tf.where(cast(out_t > tstart) * cast(out_t < tstop))

            phases = tf.gather(out_t, inds)
            phases = (phases - offset) % period
            phases = 2.0 * phases - 1.0
            phases = tf.reshape(phases, -1).numpy()
            
            indices = tf.gather(out_i, inds, axis=1)[:,:,0]
            
            for k in range(indices.shape[1]):
                ind = indices[:,k].numpy()
                all_phases[j, ind[0], ind[1], ind[2]] = phases[k]
                
            #all_phases.append(tf.scatter_nd(tf.transpose(indices), phases, shape=shape))
            

        outputs.append(tf.stack(all_phases, axis=0))

    return outputs

"""
A loss function which maximises similarity between all VSA vectors.
"""
#@tf.function
def vsa_loss(y, yh):
    loss = tf.math.reduce_mean(1 - similarity(y, yh))
    return loss