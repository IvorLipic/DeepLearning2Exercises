import math
from typing import Union, Collection

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float]

class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi
    
    @property
    def pi(self):
        return tf.nn.softmax(self.logpi)

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)

        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function
    def loss(self, data: TFData):
        l_x_z = tf.convert_to_tensor([GMModel.neglog_normal_pdf(data, self.mean[k], self.logvar[k]) for k in range(self.K)])
        l_z = tf.reduce_logsumexp([self.logpi[j] for j in range(self.K)], axis=0) - self.logpi
        return -tf.reduce_logsumexp(-(l_x_z + l_z), axis=0)

    def p_xz(self, x: TFData, k: int) -> TFData:
        return tf.exp(-0.5 * (x - self.mean[k]) ** 2 / tf.exp(self.logvar[k])) / tf.math.sqrt(2 * math.pi * tf.exp(self.logvar[k]))

    def p_x(self, x: TFData) -> TFData:
        return tf.math.reduce_sum([self.pi[i] * self.p_xz(x, i) for i in range(self.K)], axis=0)
