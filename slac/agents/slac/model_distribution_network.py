from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from slac.utils import nest_utils

tfd = tfp.distributions


class Bernoulli(tf.Module):
  def __init__(self, base_depth, name=None):
    super(Bernoulli, self).__init__(name=name)
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    logits = tf.squeeze(out, axis=-1)
    return tfd.Bernoulli(logits=logits)


class Normal(tf.Module):
  def __init__(self, base_depth, scale, name=None):
    super(Normal, self).__init__(name=name)
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., 0]
    if self.scale is None:
      assert out.shape[-1].value == 2
      scale = tf.nn.softplus(out[..., 1]) + 1e-5
    else:
      assert out.shape[-1].value == 1
      scale = self.scale
    return tfd.Normal(loc=loc, scale=scale)


class MultivariateNormalDiag(tf.Module):
  def __init__(self, base_depth, latent_size, name=None):
    super(MultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 * latent_size)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.latent_size]
    scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class Deterministic(tf.Module):
  def __init__(self, base_depth, latent_size, name=None):
    super(Deterministic, self).__init__(name=name)
    self.latent_size = latent_size
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(latent_size)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    loc = self.output_layer(out)
    return tfd.Deterministic(loc=loc)


class ConstantMultivariateNormalDiag(tf.Module):
  def __init__(self, latent_size, name=None):
    super(ConstantMultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size

  def __call__(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    scale_diag = tf.ones(shape)
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class ConstantDeterministic(tf.Module):
  def __init__(self, latent_size, name=None):
    super(ConstantDeterministic, self).__init__(name=name)
    self.latent_size = latent_size

  def __call__(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    return tfd.Deterministic(loc=loc)


class Decoder(tf.Module):
  """Probabilistic decoder for `p(x_t | z_t)`.
  """

  def __init__(self, base_depth, channels=3, scale=1.0, name=None):
    super(Decoder, self).__init__(name=name)
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(channels, 5, 2)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Compressor(tf.Module):
  """Feature extractor.
  """

  def __init__(self, base_depth, feature_size, name=None):
    super(Compressor, self).__init__(name=name)
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

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
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


@gin.configurable
class ModelDistributionNetwork(tf.Module):

  def __init__(self,
               observation_spec,
               action_spec,
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               latent1_deterministic=False,
               latent2_deterministic=False,
               model_reward=False,
               model_discount=False,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               reward_stddev=None,
               name=None):
    super(ModelDistributionNetwork, self).__init__(name=name)
    self.observation_spec = observation_spec
    self.action_spec = action_spec
    self.base_depth = base_depth
    self.latent1_size = latent1_size
    self.latent2_size = latent2_size
    self.kl_analytic = kl_analytic
    self.latent1_deterministic = latent1_deterministic
    self.latent2_deterministic = latent2_deterministic
    self.model_reward = model_reward
    self.model_discount = model_discount

    if self.latent1_deterministic:
      latent1_first_prior_distribution_ctor = ConstantDeterministic
      latent1_distribution_ctor = Deterministic
    else:
      latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
      latent1_distribution_ctor = MultivariateNormalDiag
    if self.latent2_deterministic:
      latent2_distribution_ctor = Deterministic
    else:
      latent2_distribution_ctor = MultivariateNormalDiag

    # p(z_1^1)
    self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
    # p(z_1^2 | z_1^1)
    self.latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    # p(z_{t+1}^1 | z_t^2, a_t)
    self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)

    # q(z_1^1 | x_1)
    self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    self.latent2_first_posterior = self.latent2_first_prior
    # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
    self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_posterior = self.latent2_prior

    # compresses x_t into a vector
    self.compressor = Compressor(base_depth, 8 * base_depth)
    # p(x_t | z_t^1, z_t^2)
    self.decoder = Decoder(base_depth, scale=decoder_stddev)

    if self.model_reward:
      # p(r_t | z_t^1, z_t^2, a_t, z_{t+1}^1, z_{t+1}^2)
      self.reward_predictor = Normal(8 * base_depth, scale=reward_stddev)
    else:
      self.reward_predictor = None
    if self.model_discount:
      # p(d_t | z_{t+1}^1, z_{t+1}^2)
      self.discount_predictor = Bernoulli(8 * base_depth)
    else:
      self.discount_predictor = None

  @property
  def state_size(self):
    return self.latent1_size + self.latent2_size

  def compute_loss(self, images, actions, step_types, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
    sequence_length = step_types.shape[1].value - 1

    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
        latent_posterior_samples_and_dists)
    (latent1_prior_samples, latent2_prior_samples), _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization
    (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_prior_or_posterior(
        actions, step_types, images=images[:, :1])  # for visualization. condition on first image only

    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
    latent1_first_prior_dists = self.latent1_first_prior(step_types)
    # these distributions start at t=1 and the inputs are from t-1
    latent1_after_first_prior_dists = self.latent1_prior(
        latent2_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent1_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent1_reset_masks),
        latent1_first_prior_dists,
        latent1_after_first_prior_dists)

    latent2_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent2_size])
    latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
    # these distributions start at t=1 and the last 2 inputs are from t-1
    latent2_after_first_prior_dists = self.latent2_prior(
        latent1_posterior_samples[:, 1:sequence_length+1],
        latent2_posterior_samples[:, :sequence_length],
        actions[:, :sequence_length])
    latent2_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent2_reset_masks),
        latent2_first_prior_dists,
        latent2_after_first_prior_dists)

    outputs = {}

    if self.latent1_deterministic:
      latent1_kl_divergences = 0.0
    else:
      if self.kl_analytic:
        latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
      else:
        latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                  - latent1_prior_dists.log_prob(latent1_posterior_samples))
      latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
      outputs.update({
        'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),
      })
    if self.latent2_deterministic:
      latent2_kl_divergences = 0.0
    else:
      if self.latent2_posterior == self.latent2_prior:
        latent2_kl_divergences = 0.0
      else:
        if self.kl_analytic:
          latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
        else:
          latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)
                                    - latent2_prior_dists.log_prob(latent2_posterior_samples))
        latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)
      outputs.update({
        'latent2_kl_divergence': tf.reduce_mean(latent2_kl_divergences),
      })
    if not self.latent1_deterministic or not self.latent2_deterministic:
      outputs.update({
        'kl_divergence': tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences),
      })

    likelihood_dists = self.decoder(latent1_posterior_samples, latent2_posterior_samples)
    likelihood_log_probs = likelihood_dists.log_prob(images)
    likelihood_log_probs = tf.reduce_sum(likelihood_log_probs, axis=1)
    reconstruction_error = tf.reduce_sum(tf.square(images - likelihood_dists.distribution.loc),
                                         axis=list(range(-len(likelihood_dists.event_shape), 0)))
    reconstruction_error = tf.reduce_sum(reconstruction_error, axis=1)
    outputs.update({
      'log_likelihood': tf.reduce_mean(likelihood_log_probs),
      'reconstruction_error': tf.reduce_mean(reconstruction_error),
    })

    # summed over the time dimension
    elbo = likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences

    if self.model_reward:
      reward_dists = self.reward_predictor(
          latent1_posterior_samples[:, :sequence_length],
          latent2_posterior_samples[:, :sequence_length],
          actions[:, :sequence_length],
          latent1_posterior_samples[:, 1:sequence_length + 1],
          latent2_posterior_samples[:, 1:sequence_length + 1])
      reward_valid_mask = tf.cast(tf.not_equal(step_types[:, :sequence_length], ts.StepType.LAST), tf.float32)
      reward_log_probs = reward_dists.log_prob(rewards[:, :sequence_length])
      reward_log_probs = tf.reduce_sum(reward_log_probs * reward_valid_mask, axis=1)
      reward_reconstruction_error = tf.square(rewards[:, :sequence_length] - reward_dists.loc)
      reward_reconstruction_error = tf.reduce_sum(reward_reconstruction_error * reward_valid_mask, axis=1)
      outputs.update({
        'reward_log_likelihood': tf.reduce_mean(reward_log_probs),
        'reward_reconstruction_error': tf.reduce_mean(reward_reconstruction_error),
      })
      elbo += reward_log_probs

    if self.model_discount:
      discount_dists = self.discount_predictor(
          latent1_posterior_samples[:, 1:sequence_length + 1],
          latent2_posterior_samples[:, 1:sequence_length + 1])
      discount_log_probs = discount_dists.log_prob(discounts[:, :sequence_length])
      discount_log_probs = tf.reduce_sum(discount_log_probs, axis=1)
      discount_accuracy = tf.cast(
          tf.equal(tf.cast(discount_dists.mode(), tf.float32), discounts[:, :sequence_length]), tf.float32)
      discount_accuracy = tf.reduce_sum(discount_accuracy, axis=1)
      outputs.update({
        'discount_log_likelihood': tf.reduce_mean(discount_log_probs),
        'discount_accuracy': tf.reduce_mean(discount_accuracy),
      })
      elbo += discount_log_probs

    # average over the batch dimension
    loss = -tf.reduce_mean(elbo)

    posterior_images = likelihood_dists.mean()
    prior_images = self.decoder(latent1_prior_samples, latent2_prior_samples).mean()
    conditional_prior_images = self.decoder(latent1_conditional_prior_samples, latent2_conditional_prior_samples).mean()

    outputs.update({
      'elbo': tf.reduce_mean(elbo),
      'images': images,
      'posterior_images': posterior_images,
      'prior_images': prior_images,
      'conditional_prior_images': conditional_prior_images,
    })
    return loss, outputs

  def sample_prior_or_posterior(self, actions, step_types=None, images=None):
    """Samples from the prior, except for the first time steps in which conditioning images are given."""
    if step_types is None:
      batch_size = tf.shape(actions)[0]
      sequence_length = actions.shape[1].value  # should be statically defined
      step_types = tf.fill(
          [batch_size, sequence_length + 1], ts.StepType.MID)
    else:
      sequence_length = step_types.shape[1].value - 1
      actions = actions[:, :sequence_length]
    if images is not None:
      features = self.compressor(images)

    # swap batch and time axes
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])
    if images is not None:
      features = tf.transpose(features, [1, 0, 2])

    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []
    for t in range(sequence_length + 1):
      is_conditional = images is not None and (t < images.shape[1].value)
      if t == 0:
        if is_conditional:
          latent1_dist = self.latent1_first_posterior(features[t])
        else:
          latent1_dist = self.latent1_first_prior(step_types[t])  # step_types is only used to infer batch_size
        latent1_sample = latent1_dist.sample()
        if is_conditional:
          latent2_dist = self.latent2_first_posterior(latent1_sample)
        else:
          latent2_dist = self.latent2_first_prior(latent1_sample)
        latent2_sample = latent2_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        if is_conditional:
          latent1_first_dist = self.latent1_first_posterior(features[t])
          latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
        else:
          latent1_first_dist = self.latent1_first_prior(step_types[t])
          latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
        latent1_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
        latent1_sample = latent1_dist.sample()

        if is_conditional:
          latent2_first_dist = self.latent2_first_posterior(latent1_sample)
          latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
        else:
          latent2_first_dist = self.latent2_first_prior(latent1_sample)
          latent2_dist = self.latent2_prior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist)
      latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist)
      latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)
    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

  def sample_posterior(self, images, actions, step_types, features=None):
    sequence_length = step_types.shape[1].value - 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.compressor(images)

    # swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])

    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent1_dist = self.latent1_first_posterior(features[t])
        latent1_sample = latent1_dist.sample()
        latent2_dist = self.latent2_first_posterior(latent1_sample)
        latent2_sample = latent2_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        latent1_first_dist = self.latent1_first_posterior(features[t])
        latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
        latent1_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
        latent1_sample = latent1_dist.sample()

        latent2_first_dist = self.latent2_first_posterior(latent1_sample)
        latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist)
      latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist)
      latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)
    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
