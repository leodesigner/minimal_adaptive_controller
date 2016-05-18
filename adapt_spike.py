import numpy as np


class AdaptiveSpike(object):
    def __init__(self, n_inputs, n_outputs, n_neurons,
                 seed=None, learning_rate=1e-3,
                 has_neuron_state=True, smoothing=0):
        self.rng = np.random.RandomState(seed=seed)
        self.compute_encoders(n_inputs, n_neurons)
        self.initialize_decoders(n_neurons, n_outputs)
        self.learning_rate=learning_rate
        self.has_neuron_state=has_neuron_state
        self.input_max = 1.0
        self.is_spiking = True
        self.smoothing = smoothing
        if smoothing > 0:
            self.smoothing_decay = np.exp(-1.0/smoothing)
            self.smoothing_state = np.zeros(n_outputs)

        if has_neuron_state:
            self.state = np.zeros(n_neurons)


    def step(self, state, error):
        # feed input over the static synapses
        current = self.compute_neuron_input(state)
        # do the neural nonlinearity
        activity = self.neuron(current)
        # apply the learned synapses
        value = self.compute_output(activity)

        # update the synapses with the learning rule
        index = np.where(activity>0)
        self.decoder[:,index] -= error * self.learning_rate
        #  Note: that multiply can be changed to a shift if the
        # learning_rate is a power of 2

        return value

    def compute_encoders(self, n_inputs, n_neurons):
        # generate the static synapses
        # NOTE: this algorithm could be changed, and just needs to
        # produce a similar distribution of connection weights.  Changing
        # this distribution slightly changes the class of functions the
        # neural network will be good at learning
        max_rates = self.rng.uniform(0.5, 1, n_neurons)
        intercepts = self.rng.uniform(-1, 1, n_neurons)

        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain

        enc = self.rng.randn(n_neurons, n_inputs)
        enc /= np.linalg.norm(enc, axis=1)[:,None]

        self.encoder = enc * gain[:, None]
        self.bias = bias

    def initialize_decoders(self, n_neurons, n_outputs):
        self.decoder = np.zeros((n_outputs, n_neurons))

    def compute_neuron_input(self, state):
        # there is currently still a multiply here.  But, since self.encoder is
        # randomly generated, we can replace this with any easy-to-compute
        # system.  For example, we could replace the multiplies with shifts
        # by rounding all the numbers to powers of 2.
        return np.dot(self.encoder, state) + self.bias

    def neuron(self, current):
        if self.has_neuron_state:
            # this is the accumulator implementation for a spike
            self.state = self.state + current
            self.state = np.where(self.state < 0, 0, self.state)
            spikes = np.where(self.state > 1.0, 1, 0)
            self.state[spikes>0] -= 1.0
        else:
            # this is the rng implementation for a spike
            spikes = np.where(self.rng.uniform(0, 1, len(current))<current,
                              1, 0)
        return spikes

    def compute_output(self, activity):
        decoder_access = self.decoder[:,np.where(activity>0)[0]]
        if decoder_access.shape[1]>0:
            value = np.sum(decoder_access, axis=1)
        else:
            value = np.zeros(decoder_access.shape[0])
        if self.smoothing:
            decay = self.smoothing_decay
            self.smoothing_state = (self.smoothing_state*decay +
                                    value*(1.0-decay))
            value = self.smoothing_state
        return value

