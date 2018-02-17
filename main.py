import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

import problem

TF_DTYPE = tf.float64

class DeepBSDESolver(object):

    def set(self, key, value): self._parameters[key] = value
    def get(self, key): return self._parameters[key]

    def __init__(self):

        # Default parameters
        self._parameters = {}
        self.set('batch_size', 64)
        self.set('epsilon', 1e-6)
        self.set('initial_value_minimum', 0.0)
        self.set('initial_value_maximum', 1.0)
        self.set('initial_gradient_minimum', -0.1)
        self.set('initial_gradient_maximum', 0.1)
        self.set('learning_rate', 5e-4)
        self.set('logging_frequency', 25)
        self.set('momentum', 0.99)
        self.set('number_of_epochs', 4000)
        self.set('number_of_hidden_layers', 2)
        self.set('number_of_neurons_per_hidden_layer', 110)
        self.set('number_of_samples', 256)
        self.set('number_of_time_intervals', 20)
        self.set('verbose', True)

    def run(self, session, problem):

        self._session = session
        self._problem = problem

        #########
        # BUILD #
        #########

        # We will store all additional training operations here
        self._extra_training_operations = []

        # Timestep size
        dt = self._problem.final_time / self.get('number_of_time_intervals')

        # Timesteps t_0 < t_1 < ... < t_{N-1}
        t = np.arange(0, self.get('number_of_time_intervals')) * dt

        # Placeholder for increments of Brownian motion
        self._dW = tf.placeholder(
            dtype=TF_DTYPE,
            shape=[
                None,
                self._problem.dimension,
                self.get('number_of_time_intervals')
            ]
        )

        # Placeholder for values of the state process
        self._X = tf.placeholder(
            dtype=TF_DTYPE,
            shape=[
                None,
                self._problem.dimension,
                self.get('number_of_time_intervals') + 1
            ]
        )

        # Initial guess for the value at time zero
        self._Y_0 = tf.Variable(
            initial_value=tf.random_uniform(
                shape=[1],
                minval=self.get('initial_value_minimum'),
                maxval=self.get('initial_value_maximum'),
                dtype=TF_DTYPE
            )
        )

        # Placeholder for a boolean value that determines whether or not we are training the model
        self._is_training = tf.placeholder(tf.bool)

        # Initial guess for the gradient at time zero
        Z_0 = tf.Variable(
            initial_value=tf.random_uniform(
                [1, self._problem.dimension],
                self.get('initial_gradient_minimum'),
                self.get('initial_gradient_maximum'),
                TF_DTYPE
            )
        )

        # Vector of all ones
        ones = tf.ones(
            shape=tf.stack( [tf.shape(self._dW)[0], 1] ),
            dtype=TF_DTYPE
        )

        # Initial guesses
        Y = ones * self._Y_0
        Z = tf.matmul(ones, Z_0)

        # Advance from the initial to the final time
        n = 0
        while True:
            # Y_{t_{n+1}} ~= Y_{t_n} - f dt + Z_{t_n} dW
            Y = Y \
                - self._problem.generator(t[n], self._X[:, :, n], Y, Z) * dt \
                + tf.reduce_sum(Z * self._dW[:, :, n], axis=1, keepdims=True)

            n = n + 1
            if n == self.get('number_of_time_intervals'):
                # No need to approximate Z_T, so break here
                break

            # Build network to approximate Z_{t_n}
            with tf.variable_scope('t_{}'.format(n)):
                with tf.variable_scope('layer_0'):
                    # Batch normalization
                    tmp = self._batch_normalize(self._X[:, :, n])
                for l in xrange(1, self.get('number_of_hidden_layers') + 1):
                    with tf.variable_scope('layer_{}'.format(l)):
                        # Hidden layer
                        tmp = self._layer(
                            tmp,
                            self.get('number_of_neurons_per_hidden_layer'),
                            tf.nn.relu
                        )
                # Output layer
                tmp = self._layer(
                    tmp,
                    self._problem.dimension
                )
                Z = tmp / self._problem.dimension

        # Cost function
        delta = Y - self._problem.payoff(self._X[:, :, -1])
        self._cost = tf.reduce_mean(tf.square(delta))

        # Training operations
        global_step = tf.get_variable(
            'global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(self._cost, trainable_variables)
        optimizer = tf.train.AdamOptimizer(self.get('learning_rate'))
        gradient_update = optimizer.apply_gradients(
            zip(gradients, trainable_variables),
            global_step
        )
        tmp = [gradient_update] + self._extra_training_operations
        self._training_operations = tf.group(*tmp)

        #########
        # TRAIN #
        #########

        history = []
        dW_validate, X_validate = self._problem.sample(
            self.get('number_of_samples'),
            self.get('number_of_time_intervals')
        )
        validate_dictionary = {
            self._dW: dW_validate,
            self._X: X_validate,
            self._is_training: False
        }
        self._session.run(tf.global_variables_initializer())
        for epoch in xrange(self.get('number_of_epochs') + 1):
            if epoch % self.get('logging_frequency') == 0:
                cost, Y_0 = self._session.run(
                    [self._cost, self._Y_0],
                    feed_dict=validate_dictionary
                )
                history.append([epoch, cost, Y_0])
                if self.get('verbose'):
                    logging.info(
                        'epoch: %5u   cost: %f   Y_0: %f' % (epoch, cost, Y_0)
                    )
            dW_training, X_training = self._problem.sample(
                self.get('number_of_samples'),
                self.get('number_of_time_intervals')
            )
            training_dictionary = {
                self._dW: dW_training,
                self._X: X_training,
                self._is_training: True
            }
            self._session.run(
                self._training_operations,
                feed_dict=training_dictionary
            )
        return np.array(history)

    def _layer(self, x, number_of_neurons, activation=None):
        shape = x.get_shape().as_list()
        weight = tf.get_variable(
            'Matrix',
            shape=[shape[1], number_of_neurons],
            dtype=TF_DTYPE,
            initializer=tf.random_normal_initializer(
                stddev=5.0 / np.sqrt(shape[1] + number_of_neurons)
            )
        )
        result = tf.matmul(x, weight)
        result_normalized = self._batch_normalize(result)
        if activation: return activation(result_normalized)
        else: return result_normalized

    def _batch_normalize(self, x):
        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=0.1,
                dtype=TF_DTYPE
            )
        )
        gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.random_uniform_initializer(
                minval=0.1, maxval=0.5,
                dtype=TF_DTYPE
            )
        )
        moving_mean = tf.get_variable(
            'moving_mean',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.constant_initializer(0.0, TF_DTYPE),
            trainable=False
        )
        moving_variance = tf.get_variable(
            'moving_variance',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.constant_initializer(1.0, TF_DTYPE),
            trainable=False
        )
        mean, variance = tf.nn.moments(x, [0])
        self._extra_training_operations.append(
            moving_averages.assign_moving_average(
                moving_mean,
                mean,
                self.get('momentum')
            )
        )
        self._extra_training_operations.append(
            moving_averages.assign_moving_average(
                moving_variance,
                variance,
                self.get('momentum')
            )
        )
        mean, variance = tf.cond(
            self._is_training,
            lambda: (mean, variance),
            lambda: (moving_mean, moving_variance)
        )
        result = tf.nn.batch_normalization(
            x,
            mean, variance,
            beta, gamma,
            self.get('epsilon')
        )
        result.set_shape(x.get_shape())
        return result

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-6s %(message)s'
    )

    solver = DeepBSDESolver()

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('problem_name', 'HJB', """The name of the problem.""")
    problem_name = FLAGS.problem_name

    if problem_name == 'HJB':
        problem = problem.HJB()
        solver.set('learning_rate', 1e-2)
        solver.set('number_of_epochs', 2000)
    elif problem_name == 'AllenCahn':
        problem = problem.AllenCahn()
        solver.set('learning_rate', 5e-4)
        solver.set('number_of_epochs', 4000)
    else:
        raise ValueError

    tf.reset_default_graph()
    with tf.Session() as session:
        solver.run(
            session,
            problem
        )
