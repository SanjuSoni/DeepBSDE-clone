#!/usr/bin/env python

import inspect
import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

import problem

TF_DTYPE = tf.float64

# Default parameters
DEFAULT_OPTIONS = {}
DEFAULT_OPTIONS['apply_batch_normalization'] = True
DEFAULT_OPTIONS['epsilon'] = 1e-6
DEFAULT_OPTIONS['initial_beta_std'] = 1.
DEFAULT_OPTIONS['initial_gamma_minimum'] = 0.
DEFAULT_OPTIONS['initial_gamma_maximum'] = 1.
DEFAULT_OPTIONS['initial_gradient_minimum'] = -.1
DEFAULT_OPTIONS['initial_gradient_maximum'] = .1
DEFAULT_OPTIONS['initial_value_minimum'] = 0.
DEFAULT_OPTIONS['initial_value_maximum'] = 1.
DEFAULT_OPTIONS['learning_rates'] = [1e-2]
DEFAULT_OPTIONS['learning_rate_boundaries'] = []
DEFAULT_OPTIONS['logging_frequency'] = 25
DEFAULT_OPTIONS['momentum'] = 0.99
DEFAULT_OPTIONS['number_of_epochs'] = 2000
DEFAULT_OPTIONS['number_of_hidden_layers'] = 4
DEFAULT_OPTIONS['number_of_neurons_per_hidden_layer'] = None # default: dim + 10
DEFAULT_OPTIONS['number_of_test_samples'] = 256
DEFAULT_OPTIONS['number_of_time_intervals'] = 20
DEFAULT_OPTIONS['number_of_training_samples'] = 256
DEFAULT_OPTIONS['produce_summary'] = True

def solve(problem, session=tf.Session(), **kwargs):

    # Dictionary of options
    options = DEFAULT_OPTIONS.copy()
    options.update(kwargs)

    # We will store all additional training operations here
    extra_training_operations = []

    # Placeholder for a boolean value that determines whether or not we are
    # training the model
    is_training = tf.placeholder(tf.bool)

    def _layer(inputs, number_of_neurons, activation=None):

        result = tf.contrib.layers.fully_connected(
            inputs,
            number_of_neurons,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False
            )
        )

        if options['apply_batch_normalization']:
            shape = result.get_shape()
            beta = tf.get_variable(
                'beta',
                shape=[shape[-1]],
                dtype=TF_DTYPE,
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=options['initial_beta_std'],
                    dtype=TF_DTYPE
                )
            )
            gamma = tf.get_variable(
                'gamma',
                shape=[shape[-1]],
                dtype=TF_DTYPE,
                initializer=tf.random_uniform_initializer(
                    minval=options['initial_gamma_minimum'],
                    maxval=options['initial_gamma_maximum'],
                    dtype=TF_DTYPE
                )
            )
            moving_mean = tf.get_variable(
                'moving_mean',
                shape=[shape[-1]],
                dtype=TF_DTYPE,
                initializer=tf.constant_initializer(0.0, TF_DTYPE),
                trainable=False
            )
            moving_variance = tf.get_variable(
                'moving_variance',
                shape=[shape[-1]],
                dtype=TF_DTYPE,
                initializer=tf.constant_initializer(1.0, TF_DTYPE),
                trainable=False
            )
            mean, variance = tf.nn.moments(result, [0])
            extra_training_operations.append(
                moving_averages.assign_moving_average(
                    moving_mean,
                    mean,
                    options['momentum']
                )
            )
            extra_training_operations.append(
                moving_averages.assign_moving_average(
                    moving_variance,
                    variance,
                    options['momentum']
                )
            )
            mean, variance = tf.cond(
                is_training,
                lambda: (mean, variance),
                lambda: (moving_mean, moving_variance)
            )
            result = tf.nn.batch_normalization(
                result,
                mean, variance,
                beta, gamma,
                options['epsilon']
            )
            result.set_shape(shape)

        if activation:
            return activation(result)
        return result

    #########
    # BUILD #
    #########

    logging.debug('Building network...')

    # Timestep size
    dt = problem.final_time / options['number_of_time_intervals']

    # Timesteps t_0 < t_1 < ... < t_{N-1}
    t = np.arange(0, options['number_of_time_intervals']) * dt

    # Placeholder for increments of Brownian motion
    dW = tf.placeholder(
        dtype=TF_DTYPE,
        shape=[
            None,
            problem.dimension,
            options['number_of_time_intervals']
        ]
    )

    # Placeholder for values of the state process
    X = tf.placeholder(
        dtype=TF_DTYPE,
        shape=[
            None,
            problem.dimension,
            options['number_of_time_intervals'] + 1
        ]
    )

    # Initial guess for the value at time zero
    Y_0 = tf.get_variable(
        'Y_0',
        shape=[],
        dtype=TF_DTYPE,
        initializer=tf.random_uniform_initializer(
            minval=options['initial_value_minimum'],
            maxval=options['initial_value_maximum'],
            dtype=TF_DTYPE
        )
    )
    if options['produce_summary']:
        tf.summary.scalar('Y_0', Y_0)

    # Initial guess for the gradient at time zero
    Z_0 = tf.get_variable(
        'Z_0',
        shape=[1, problem.dimension],
        dtype=TF_DTYPE,
        initializer=tf.random_uniform_initializer(
            minval=options['initial_gradient_minimum'],
            maxval=options['initial_gradient_maximum'],
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
    number_of_neurons = options['number_of_neurons_per_hidden_layer']
    if number_of_neurons is None:
        number_of_neurons = problem.dimension + 10

    # Advance from the initial to the final time
    n = 0
    while True:
        # Y_{t_{n+1}} ~= Y_{t_n} - f dt + Z_{t_n} dW
        Y = Y \
            - problem.generator(t[n], X[:, :, n], Y, Z) * dt \
            + tf.reduce_sum(Z * dW[:, :, n], axis=1, keepdims=True)
        if options['produce_summary']:
            tf.summary.scalar('E_Y_{}'.format(n+1), tf.reduce_mean(Y))

        n = n + 1
        if n == options['number_of_time_intervals']:
            # No need to approximate Z_T, so break here
            break

        # Build network to approximate Z_{t_n}
        with tf.variable_scope('t_{}'.format(n)):
            tmp = X[:, :, n]

            # Hidden layers
            for l in xrange(1, options['number_of_hidden_layers'] + 1):
                with tf.variable_scope('layer_{}'.format(l)):
                    tmp = _layer(
                        tmp,
                        number_of_neurons,
                        tf.nn.relu,
                    )

            # Output layer
            tmp = _layer(
                tmp,
                problem.dimension
            )
            Z = tmp / problem.dimension

    # Cost function
    delta = Y - problem.terminal(X[:, :, -1])
    loss = tf.reduce_mean(tf.square(delta))
    if options['produce_summary']:
        tf.summary.scalar('loss', loss)

    # Training operations
    global_step = tf.get_variable(
        'global_step',
        shape=[],
        dtype=tf.int32,
        initializer=tf.constant_initializer(0),
        trainable=False
    )
    learning_rates = options['learning_rates']
    learning_rate_boundaries = options['learning_rate_boundaries']
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
    gradients = tf.gradients(loss, trainable_variables)
    optimizer = tf.train.AdamOptimizer(learning_rate_function)
    gradient_update = optimizer.apply_gradients(
        zip(gradients, trainable_variables),
        global_step
    )
    tmp = [gradient_update] + extra_training_operations
    training_operations = tf.group(*tmp)

    if options['produce_summary']:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            FLAGS.summaries_directory,
            session.graph
        )

    #########
    # TRAIN #
    #########

    logging.debug('Training network...')

    dW_test, X_test = problem.sample(
        options['number_of_test_samples'],
        options['number_of_time_intervals']
    )
    validate_dictionary = {
        dW: dW_test,
        X :  X_test,
        is_training: False
    }
    session.run(tf.global_variables_initializer())
    for epoch in xrange(options['number_of_epochs'] + 1):
        if epoch % options['logging_frequency'] == 0:
            observed_loss, observed_Y_0 = session.run(
                [loss, Y_0],
                feed_dict=validate_dictionary
            )
            logging.info(
                'epoch: %5u   loss: %8f   Y_0: %8f'
                % (epoch, observed_loss, observed_Y_0)
            )
        dW_training, X_training = problem.sample(
            options['number_of_training_samples'],
            options['number_of_time_intervals']
        )
        training_dictionary = {
            dW: dW_training,
            X :  X_training,
            is_training: True
        }
        if options['produce_summary']:
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

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)-6s %(message)s'
    )

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'problem_name',
        '',
        'Name of problem to solve'
    )
    tf.app.flags.DEFINE_string(
        'summaries_directory',
        '/tmp/deep-bsde',
        'Where to store summaries'
    )

    # Select the problem
    problem = None
    try:
        if FLAGS.problem_name == 'Problem': raise AttributeError
        problem = getattr(sys.modules['problem'], FLAGS.problem_name)()
    except AttributeError:
        problem_names = [name for name, _ in inspect.getmembers(
            sys.modules['problem'],
            lambda member: inspect.isclass(member)
        ) if name != 'Problem']
        print('usage: python deep-bsde.py --problem_name=PROBLEM_NAME [--summaries_directory=PATH]')
        print('PROBLEM_NAME is one of %s' % ', '.join(problem_names))
    else:
        solve(problem)
