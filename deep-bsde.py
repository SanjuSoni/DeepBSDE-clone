#!/usr/bin/env python

import inspect
import logging
import sys

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
        self.set('apply_batch_normalization', True)
        self.set('epsilon', 1e-6)
        self.set('initial_beta_std', 1.)
        self.set('initial_gamma_minimum', 0.)
        self.set('initial_gamma_maximum', 1.)
        self.set('initial_gradient_minimum', -.1)
        self.set('initial_gradient_maximum', .1)
        self.set('initial_value_minimum', 0.)
        self.set('initial_value_maximum', 1.)
        self.set('learning_rates', [1e-2])
        self.set('learning_rate_boundaries', [])
        self.set('logging_frequency', 25)
        self.set('momentum', 0.99)
        self.set('number_of_epochs', 2000)
        self.set('number_of_hidden_layers', 4)
        self.set('number_of_neurons_per_hidden_layer', None) # default: dim + 10
        self.set('number_of_training_samples', 256)
        self.set('number_of_test_samples', 256)
        self.set('number_of_time_intervals', 20)
        self.set('produce_summary', True)
        self.set('verbose', True)

    def run(self, session, problem):

        #########
        # BUILD #
        #########

        logging.debug('Building network...')

        # We will store all additional training operations here
        extra_training_operations = []

        # Timestep size
        dt = problem.final_time / self.get('number_of_time_intervals')

        # Timesteps t_0 < t_1 < ... < t_{N-1}
        t = np.arange(0, self.get('number_of_time_intervals')) * dt

        # Placeholder for increments of Brownian motion
        dW = tf.placeholder(
            dtype=TF_DTYPE,
            shape=[
                None,
                problem.dimension,
                self.get('number_of_time_intervals')
            ]
        )

        # Placeholder for values of the state process
        X = tf.placeholder(
            dtype=TF_DTYPE,
            shape=[
                None,
                problem.dimension,
                self.get('number_of_time_intervals') + 1
            ]
        )

        # Initial guess for the value at time zero
        Y_0 = tf.get_variable(
            'Y_0',
            shape=[],
            dtype=TF_DTYPE,
            initializer=tf.random_uniform_initializer(
                minval=self.get('initial_value_minimum'),
                maxval=self.get('initial_value_maximum'),
                dtype=TF_DTYPE
            )
        )
        if self.get('produce_summary'):
            tf.summary.scalar('Y_0', Y_0)

        # Placeholder for a boolean value that determines whether or not we are
        # training the model
        is_training = tf.placeholder(tf.bool)

        # Initial guess for the gradient at time zero
        Z_0 = tf.get_variable(
            'Z_0',
            shape=[1, problem.dimension],
            dtype=TF_DTYPE,
            initializer=tf.random_uniform_initializer(
                minval=self.get('initial_gradient_minimum'),
                maxval=self.get('initial_gradient_maximum'),
                dtype=TF_DTYPE
            )
        )

        # Vector of all ones
        ones = tf.ones(
            shape=tf.stack( [tf.shape(dW)[0], 1] ),
            dtype=TF_DTYPE
        )

        # Initial guesses
        Y = ones * Y_0
        Z = tf.matmul(ones, Z_0)

        # Number of neurons per hidden layer
        number_of_neurons = self.get('number_of_neurons_per_hidden_layer')
        if number_of_neurons is None:
            number_of_neurons = problem.dimension + 10

        # Advance from the initial to the final time
        n = 0
        while True:
            # Y_{t_{n+1}} ~= Y_{t_n} - f dt + Z_{t_n} dW
            Y = Y \
                - problem.generator(t[n], X[:, :, n], Y, Z) * dt \
                + tf.reduce_sum(Z * dW[:, :, n], axis=1, keepdims=True)
            if self.get('produce_summary'):
                tf.summary.scalar('E_Y_{}'.format(n+1), tf.reduce_mean(Y))

            n = n + 1
            if n == self.get('number_of_time_intervals'):
                # No need to approximate Z_T, so break here
                break

            # Build network to approximate Z_{t_n}
            with tf.variable_scope('t_{}'.format(n)):
                tmp = X[:, :, n]

                # Hidden layers
                for l in xrange(1, self.get('number_of_hidden_layers') + 1):
                    with tf.variable_scope('layer_{}'.format(l)):
                        tmp = self._layer(
                            tmp,
                            number_of_neurons,
                            tf.nn.relu,
                            is_training,
                            extra_training_operations
                        )

                # Output layer
                tmp = self._layer(
                    tmp,
                    problem.dimension,
                    None,
                    is_training,
                    extra_training_operations
                )
                Z = tmp / problem.dimension

        # Cost function
        delta = Y - problem.terminal(X[:, :, -1])
        cost = tf.reduce_mean(tf.square(delta))
        if self.get('produce_summary'):
            tf.summary.scalar('cost', cost)

        # Training operations
        global_step = tf.get_variable(
            'global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        learning_rates = self.get('learning_rates')
        learning_rate_boundaries = self.get('learning_rate_boundaries')
        assert(len(learning_rates) == len(learning_rate_boundaries)+1)
        if len(learning_rates) == 1:
            learning_rate_function = learning_rates[0]
        else:
            learning_rate_function = tf.train.piecewise_constant(
                global_step,
                learning_rate_boundaries,
                learning_rates
            )
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(cost, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate_function)
        gradient_update = optimizer.apply_gradients(
            zip(gradients, trainable_variables),
            global_step
        )
        tmp = [gradient_update] + extra_training_operations
        training_operations = tf.group(*tmp)

        if self.get('produce_summary'):
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                FLAGS.summaries_dir,
                session.graph
            )

        #########
        # TRAIN #
        #########

        logging.debug('Training network...')

        dW_test, X_test = problem.sample(
            self.get('number_of_test_samples'),
            self.get('number_of_time_intervals')
        )
        validate_dictionary = {
            dW: dW_test,
            X :  X_test,
            is_training: False
        }
        session.run(tf.global_variables_initializer())
        for epoch in xrange(self.get('number_of_epochs') + 1):
            if epoch % self.get('logging_frequency') == 0:
                observed_cost, observed_Y_0 = session.run(
                    [cost, Y_0],
                    feed_dict=validate_dictionary
                )
                if self.get('verbose'):
                    logging.info(
                        'epoch: %5u   cost: %8f   Y_0: %8f'
                        % (epoch, observed_cost, observed_Y_0)
                    )
            dW_training, X_training = problem.sample(
                self.get('number_of_training_samples'),
                self.get('number_of_time_intervals')
            )
            training_dictionary = {
                dW: dW_training,
                X :  X_training,
                is_training: True
            }
            if self.get('produce_summary'):
                summary, _ = session.run(
                    [merged, training_operations],
                    feed_dict=training_dictionary
                )
                train_writer.add_summary(summary, epoch)
            else:
                session.run(
                    training_operations,
                    feed_dict=training_dictionary
                )

    def _layer(self, x, number_of_neurons, activation, is_training, ops):

        result = tf.contrib.layers.fully_connected(
            x,
            number_of_neurons,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False
            )
        )

        if self.get('apply_batch_normalization'):
            result = self._batch_normalize(result, is_training, ops)

        if activation:
            return activation(result)

        return result

    def _batch_normalize(self, x, is_training, ops):

        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.get('initial_beta_std'),
                dtype=TF_DTYPE
            )
        )
        gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            dtype=TF_DTYPE,
            initializer=tf.random_uniform_initializer(
                minval=self.get('initial_gamma_minimum'),
                maxval=self.get('initial_gamma_maximum'),
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
        ops.append(
            moving_averages.assign_moving_average(
                moving_mean,
                mean,
                self.get('momentum')
            )
        )
        ops.append(
            moving_averages.assign_moving_average(
                moving_variance,
                variance,
                self.get('momentum')
            )
        )
        mean, variance = tf.cond(
            is_training,
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
        level=logging.DEBUG,
        format='%(levelname)-6s %(message)s'
    )

    # Select the problem
    problem = None
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('problem_name', '', 'Name of problem to solve')
    tf.app.flags.DEFINE_string('summaries_dir', '/tmp/deep-bsde', 'Where to store summaries')
    try:
        if FLAGS.problem_name == 'Problem': raise AttributeError
        problem = getattr(sys.modules['problem'], FLAGS.problem_name)()
    except AttributeError:
        problem_names = [name for name, _ in inspect.getmembers(
            sys.modules['problem'],
            lambda member: inspect.isclass(member)
        ) if name != 'Problem']
        print('usage: python deep-bsde.py --problem_name=PROBLEM_NAME')
        print('PROBLEM_NAME is one of %s' % ', '.join(problem_names))
    else:
        session = tf.Session()
        DeepBSDESolver().run(
            session,
            problem
        )
