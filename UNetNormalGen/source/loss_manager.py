import tensorflow as tf
import math

class LossManager(object):

    def __init__(self, labels, last_layer):
        super(LossManager, self).__init__()
        self._loss = self.cosine_similarity
        self._labels = labels
        self._last_layer = last_layer

    def cosine_similarity(self):
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=3, reduction=tf.losses.Reduction.NONE)
        loss = cosine_loss(self._labels, self._last_layer)
        loss_flattened = tf.layers.Flatten()(loss)
        LossManager.angels_precentage(tf.math.acos(loss_flattened))
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Cosine Angle', loss)
        tf.summary.scalar('Cosine distance', 1-loss)
        return 1-loss

    @staticmethod
    def angels_precentage(angels_diff):
        angels_diff = angels_diff * 180.0 / math.pi
        less = tf.cast(tf.math.less(angels_diff, 30.0), tf.float32)
        sum_ = tf.math.reduce_mean(less)
        tf.summary.scalar('sum 30', sum_)
        less = tf.cast(tf.math.less(angels_diff, 22.5), tf.float32)
        sum_ = tf.math.reduce_mean(less)
        tf.summary.scalar('sum 22.5', sum_)
        less = tf.cast(tf.math.less(angels_diff, 11.5), tf.float32)
        sum_ = tf.math.reduce_mean(less)
        tf.summary.scalar('sum 11.5', sum_)

