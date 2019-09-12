from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
nest = tf.contrib.framework.nest


class SoftlearningPreprocessor(tf.Module):
  def __init__(self, base_depth, n_layers=2, name=None):
    super(SoftlearningPreprocessor, self).__init__(name=name)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.relu)
    pool = tf.keras.layers.MaxPool2D
    self.convs = []
    self.pools = []
    spatial_size = 64
    for i in range(n_layers):
      self.convs.append(conv(base_depth, 3, 1))
      self.pools.append(pool(2, 2))
      spatial_size //= 2
    self.feature_size = spatial_size * spatial_size * base_depth

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    for conv, pool in zip(self.convs, self.pools):
      out = conv(out)
      out = pool(out)
    out = tf.keras.layers.Flatten()(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [-1]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, hidden)


class Compressor(tf.Module):
  """Feature extractor.
  """
  def __init__(self, base_depth, name=None):
    super(Compressor, self).__init__(name=name)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")
    self.feature_size = 8 * base_depth

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, hidden)


class D4pgPreprocessor(tf.Module):
  """Feature extractor.
  """
  def __init__(self, name=None):
    super(D4pgPreprocessor, self).__init__(name=name)
    self.conv_layers = [
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="VALID",
            activation=tf.keras.activations.elu,
            kernel_initializer=tf.keras.initializers.glorot_uniform()),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="VALID",
            activation=tf.keras.activations.elu,
            kernel_initializer=tf.keras.initializers.glorot_uniform())
    ]
    spatial_size = 29
    self.feature_size = spatial_size * spatial_size * 32

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    for layer in self.conv_layers:
      out = layer(out)
    out = tf.keras.layers.Flatten()(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, hidden)


class Preprocessor(tf.Module):
  """Feature extractor.
  """
  def __init__(self, filters, n_layers, name=None):
    import IPython as ipy; ipy.embed();
    super(Preprocessor, self).__init__(name=name)
    self.conv_layers = [
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        for _ in range(n_layers)
    ]
    spatial_size = 64
    for i in range(n_layers):
      spatial_size //= 2
    self.feature_size = spatial_size * spatial_size * filters

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    for layer in self.conv_layers:
      out = layer(out)
    out = tf.keras.layers.Flatten()(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, hidden)
