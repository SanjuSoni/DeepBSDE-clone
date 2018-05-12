import numpy as np
import tensorflow as tf

class Problem(object):
    def __init__(self, dimension, final_time):
        self._dimension = dimension
        self._final_time = final_time

    @property
    def dimension(self): return self._dimension

    @property
    def final_time(self): return self._final_time

    def sample(self, number_of_samples, number_of_time_intervals): raise NotImplementedError
    def generator(self, t, X_t, Y_t, Z_t): raise NotImplementedError
    def terminal(self, X_t): raise NotImplementedError

class PutOnMin(Problem):
    def __init__(self, dimension=1, final_time=1.0, initial_price=100., strike=100., interest_rate=0.04, volatility=0.2):
        super(PutOnMin, self).__init__(dimension, final_time)
        self._X_0 = initial_price
        self._K = strike
        self._r = interest_rate
        self._sigma = volatility

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.final_time / number_of_time_intervals
        dW = np.random.normal(
            size=[number_of_samples, self.dimension, number_of_time_intervals]
        ) * np.sqrt(dt)
        X = self._X_0 * np.ones([number_of_samples, self.dimension, number_of_time_intervals + 1])
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = X[:, :, n] + self._r * X[:, :, n] * dt + self._sigma * X[:, :, n] * dW[:, :, n]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        return 0.

    def terminal(self, X_T):
        return np.exp(-self._r * self.final_time) * tf.maximum(
            self._K - tf.reduce_min(X_T, axis=1, keepdims=True),
            0.
        )

class AllenCahn(Problem):
    def __init__(self, dimension=100, final_time=.3):
        super(AllenCahn, self).__init__(dimension, final_time)
        self._sigma = np.sqrt(2.)

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.final_time / number_of_time_intervals
        dW = np.random.normal(
            size=[number_of_samples, self.dimension, number_of_time_intervals]
        ) * np.sqrt(dt)
        X = np.zeros([number_of_samples, self.dimension, number_of_time_intervals + 1])
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = X[:, :, n] + self._sigma * dW[:, :, n]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        return Y_t - tf.pow(Y_t, 3)

    def terminal(self, X_T):
        return .5 / (1. + .2 * tf.reduce_sum(tf.square(X_T), axis=1, keepdims=True))

class HJB(Problem):
    def __init__(self, dimension=100, final_time=1.0):
        super(HJB, self).__init__(dimension, final_time)
        self._sigma = np.sqrt(2.)
        self._lambda = 1.

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.final_time / number_of_time_intervals
        dW = np.random.normal(
            size=[number_of_samples, self.dimension, number_of_time_intervals]
        ) * np.sqrt(dt)
        X = np.zeros([number_of_samples, self.dimension, number_of_time_intervals + 1])
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = X[:, :, n] + self._sigma * dW[:, :, n]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        return -self._lambda * tf.reduce_sum(tf.square(Z_t), axis=1, keepdims=True)

    def terminal(self, X_T):
        return tf.log((1. + tf.reduce_sum(tf.square(X_T), axis=1, keepdims=True)) / 2.)
