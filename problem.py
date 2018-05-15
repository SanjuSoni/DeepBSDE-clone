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

    def sample(self, number_of_samples, number_of_time_intervals):
        raise NotImplementedError

    def generator(self, t, X_t, Y_t, Z_t):
        raise NotImplementedError

    def terminal(self, X_t):
        raise NotImplementedError

class PutOnMin(Problem):
    def __init__(
        self,
        dimension=1,
        final_time=1.,
        initial_price=100.,
        strike=100.,
        interest_rate=.04,
        volatility=.2
    ):
        super(PutOnMin, self).__init__(dimension, final_time)
        self._X_0 = initial_price
        self._K = strike
        self._r = interest_rate
        self._sigma = volatility

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.final_time / number_of_time_intervals
        dW = np.random.normal(size=[
            number_of_samples,
            self.dimension,
            number_of_time_intervals
        ]) * np.sqrt(dt)
        X = np.empty([
            number_of_samples,
            self.dimension,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = self._X_0
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = (1. + self._r * dt + self._sigma * dW[:, :, n]) \
                         * X[:, :, n]
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
        dW = np.random.normal(size=[
            number_of_samples,
            self.dimension,
            number_of_time_intervals
        ]) * np.sqrt(dt)
        X = np.empty([
            number_of_samples,
            self.dimension,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = 0.
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = X[:, :, n] + self._sigma * dW[:, :, n]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        return Y_t - tf.pow(Y_t, 3)

    def terminal(self, X_T):
        tmp = tf.reduce_sum(tf.square(X_T), axis=1, keepdims=True)
        return .5 / (1. + .2 * tmp)

class HJB(Problem):
    def __init__(self, dimension=100, final_time=1.0):
        super(HJB, self).__init__(dimension, final_time)
        self._sigma = np.sqrt(2.)
        self._lambda = 1.

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.final_time / number_of_time_intervals
        dW = np.random.normal(size=[
            number_of_samples,
            self.dimension,
            number_of_time_intervals
        ]) * np.sqrt(dt)
        X = np.empty([
            number_of_samples,
            self.dimension,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = 0.
        for n in xrange(number_of_time_intervals):
            X[:, :, n+1] = X[:, :, n] + self._sigma * dW[:, :, n]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        tmp = tf.reduce_sum(tf.square(Z_t), axis=1, keepdims=True)
        return -self._lambda * tmp

    def terminal(self, X_T):
        tmp = tf.reduce_sum(tf.square(X_T), axis=1, keepdims=True)
        return tf.log((1. + tmp) / 2.)
