from keras import backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
from keras.layers import BatchNormalization, Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

import numpy as np
from numpy.random import standard_normal

# tf debugger
'''
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)
'''

################
## Parameters ##
################

# Terminal value
#g = lambda x : np.log(0.5 * (1.0 + np.sum(x ** 2, axis=-1, keepdims=True)))
g = lambda x : 1.0 / (2.0 + 2.0 / 5.0 * np.sum(x ** 2, axis=-1, keepdims=True))

d = 100                   # dimension
T = 0.3                   # final time
sigma = np.sqrt(2)        # diffusion

N = 20                    # number of timesteps
sample_size = 256         # number of random samples
neurons = d + 10          # number of neurons in hidden layers
learning_rate = 5e-4      # learning rate for optimizer
epochs = 2000             # number of training epochs
batch_size = 64           # mini-batch size

visualize = False         # Output graph of model

###########
## Logic ##
###########

inputs = []

class ValueLayer(Layer):
    def __init__(self, output_dim, minval, maxval, **kwargs):
        self.output_dim = output_dim
        self.minval = minval
        self.maxval = maxval
        super(ValueLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.value = self.add_weight( \
            name='value', \
            shape=(1, self.output_dim), \
            initializer=RandomUniform(minval=self.minval, maxval=self.maxval), \
            trainable=True \
        )
        super(ValueLayer, self).build(input_shape)
    def call(self, x):
        return self.value
    def compute_output_shape(self, input_shape):
        return (1, self.output_dim)

# Dummy input for initial solution and gradient
dummy = Input(shape=(1,), name='dummy')
inputs.append(dummy)
u_0 = ValueLayer(1, 0.3, 0.6, name='u_0')(dummy)
grad_0 = ValueLayer(d, -0.1, 0.1, name='grad_0')(dummy)

# Timestep size
dt = T / N

# Timesteps t_1 to t_{N - 1}
u_prev, grad_prev = (u_0, grad_0)
for n in range(1, N + 1):

    # Standard normal random draw
    phi_n = Input(shape=(d,), name='phi_%d' % (n))
    inputs.append(phi_n)

    # u_{t_n} = u_{t_n - 1} - f * dt + grad_n^T * sigma * (W_{t_n} - W_{t_n - 1})
    u_n = Lambda(lambda args: \
        args[0] \
        #+ K.sum(args[1] ** 2, axis=-1, keepdims=True) * dt \
        - (args[0] - args[0] ** 3) * dt \
        + sigma * K.sum(args[1] * args[2], axis=-1, keepdims=True) * np.sqrt(dt) \
    , name='u_%d' % (n))([u_prev, grad_prev, phi_n])
    u_prev = u_n

    # No need to advance state process on last iteration
    if n == N: break

    # X_{t_n} = X_{t_n - 1} + sigma * (W_{t_n} - W_{t_n - 1})
    if n > 1:
        X_n = Lambda(lambda args: \
            args[0] + sigma * args[1] * np.sqrt(dt) \
        , name='X_%d' % (n))([X_prev, phi_n])
    else:
        X_n = Lambda(lambda phi: \
            sigma * phi * np.sqrt(dt) \
        , name='X_%d' % (n))(phi_n)

    # X_n -> RELU -> RELU -> grad_n subnetwork
    X_n_normalized = BatchNormalization()(X_n)
    hidden1 = Dense(neurons, activation='relu', name='hiddenA_%d' % (n))(X_n_normalized)
    hidden2 = Dense(neurons, activation='relu', name='hiddenB_%d' % (n))(hidden1)
    grad_n = Dense(d, activation='linear', name='grad_%d' % (n))(hidden2)

    X_prev, grad_prev = (X_n, grad_n)

# Build model
model = Model(inputs=inputs, outputs=u_prev)

# Adam optimizer
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Visualization
if visualize:
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')

# Training data
dummy_variable = np.zeros((sample_size, 1))
training_inputs = [dummy_variable]
X = np.zeros((sample_size, d))
for n in range(1, N + 1):
    phi = standard_normal((sample_size, d))
    X = X + sigma * phi * np.sqrt(dt)
    training_inputs.append(phi)
training_outputs = g(X)

# Callbacks (just used to print debugging information)
class TrainingInformation(Callback):
    def __init__(self, interval):
        self._interval = interval
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self._interval == 0:
            loss = logs.get('loss')
            value = model.get_layer(name='u_0').get_weights()[0]
            print('epoch %d/%d\tY_0 = %f\tloss = %f' % (epoch, epochs, value, loss))
        return
training_information = TrainingInformation(10)

# Fit the model
model.fit( \
    training_inputs, training_outputs, \
    epochs=epochs, batch_size=batch_size, \
    callbacks=[training_information], verbose=False \
)
