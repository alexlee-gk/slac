from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils


@gin.configurable
class CriticNetwork(network.Network):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=None,
               action_fc_layer_params=None,
               joint_fc_layer_params=(256, 256),
               activation_fn=tf.nn.relu,
               name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      observation_conv_layer_params: Optional list of convolution layer
        parameters for observations, where each item is a length-three tuple
        indicating (num_units, kernel_size, stride).
      observation_fc_layer_params: Optional list of fully connected parameters
        for observations, where each item is the number of units in the layer.
      action_fc_layer_params: Optional list of fully connected parameters for
        actions, where each item is the number of units in the layer.
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        spec.
    """
    super(CriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(observation_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    self._observation_layers = utils.mlp_layers(
        observation_conv_layer_params,
        observation_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        name='observation_encoding')

    self._action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        name='action_encoding')

    self._joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        name='joint_mlp')

    self._joint_layers.append(
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            name='value'))

  def call(self, inputs, step_type=(), network_state=()):
    observation, action = inputs
    del step_type  # unused.
    observation = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
    for layer in self._observation_layers:
      observation = layer(observation)

    action = tf.cast(tf.nest.flatten(action)[0], tf.float32)
    for layer in self._action_layers:
      action = layer(action)

    joint = tf.concat([observation, action], 1)
    for layer in self._joint_layers:
      joint = layer(joint)

    return tf.reshape(joint, [-1]), network_state
