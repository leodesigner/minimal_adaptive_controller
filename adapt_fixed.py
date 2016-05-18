import numpy as np


class AdaptiveFixed(object):
    def __init__(self, n_inputs, n_outputs, n_neurons,
                 seed=None, learning_rate=1e-3,
                 has_neuron_state=True, smoothing=0):
        self.rng = np.random.RandomState(seed=seed)
        self.compute_encoders(n_inputs, n_neurons)
        self.initialize_decoders(n_neurons, n_outputs)
        self.learning_rate_shift=int(round(np.log2(1.0/learning_rate)))
        self.has_neuron_state=has_neuron_state
        self.input_max = 1<<16
        self.is_spiking = True
        if has_neuron_state:
            self.state = np.zeros(n_neurons, dtype='int64')
        self.smoothing = smoothing
        if smoothing > 0:
            smoothing_decay = np.exp(-1.0/smoothing)
            self.smoothing_shift = -int(np.round(np.log2(1-smoothing_decay)))
            self.smoothing_state = np.zeros(n_outputs, dtype='int64')

    def step(self, state, error):
        state = np.array(state, dtype='int64')
        error = np.array(error, dtype='int64')
        # feed input over the static synapses
        current = self.compute_neuron_input(state)
        # do the neural nonlinearity
        activity = self.neuron(current)
        # apply the learned synapses
        value = self.compute_output(activity)

        # update the synapses with the learning rule
        index = np.where(activity>0)
        self.decoder[:,index] -= error >> self.learning_rate_shift

        return value

    def compute_encoders(self, n_inputs, n_neurons):
        # generate the static synapses
        # NOTE: this algorithm could be changed, and just needs to produce a
        # similar distribution of connection weights.  Changing this
        # distribution slightly changes the class of functions the neural
        # network will be good at learning
        max_rates = self.rng.uniform(0.5, 1, n_neurons)
        intercepts = self.rng.uniform(-1, 1, n_neurons)

        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain

        enc = self.rng.randn(n_neurons, n_inputs)
        enc /= np.linalg.norm(enc, axis=1)[:,None]

        encoder = enc * gain[:, None]
        self.bias = (bias*(1<<16)).astype('int32')

        # store sign and shift rather than the encoder
        self.sign = np.where(encoder>0, 1, -1)
        self.shift1 = np.log2(encoder*(1<<16)*self.sign).astype(int)


    def initialize_decoders(self, n_neurons, n_outputs):
        self.decoder = np.zeros((n_outputs, n_neurons), dtype='int64')

    def compute_neuron_input(self, state):
        # this should be able to be reduced to 32 bits (or even 16)
        result = self.bias.astype('int64')<<16
        for i, s in enumerate(state):
            result += (self.sign[:,i]*(s.astype('int64')<<(self.shift1[:,i])))
        return result>>16
        # the above code approximates the following multiply using shifts
        #return np.dot(self.encoder, state) + self.bias

    def neuron(self, current):
        if self.has_neuron_state:
            # this is the accumulator implementation for a spike
            self.state = self.state + current
            self.state = np.where(self.state<0, 0, self.state)
            spikes = np.where(self.state>=(1<<16), 1, 0)
            self.state[spikes>0] -= 1<<16
        else:
            # this is the rng implementation for a spike
            spikes = np.where(self.rng.randint(0,1<<16,len(current))<current,
                              1, 0)
        return spikes

    def compute_output(self, activity):
        decoder_access = self.decoder[:,np.where(activity>0)[0]]
        if decoder_access.shape[1]>0:
            value = np.sum(decoder_access, axis=1)
        else:
            value = np.zeros(decoder_access.shape[0], dtype=int)
        if self.smoothing:
            dv = (value - self.smoothing_state) >> self.smoothing_shift
            self.smoothing_state += dv
            value = self.smoothing_state
        return value.copy()

