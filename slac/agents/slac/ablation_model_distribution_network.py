from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
from slac.agents.slac.model_distribution_network import Bernoulli
from slac.agents.slac.model_distribution_network import Compressor
from slac.agents.slac.model_distribution_network import ConstantMultivariateNormalDiag
from slac.agents.slac.model_distribution_network import Decoder
from slac.agents.slac.model_distribution_network import MultivariateNormalDiag
from slac.agents.slac.model_distribution_network import Normal
from slac.utils import nest_utils
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


@gin.configurable
class SlacModelDistributionNetwork(tf.Module):
  """Equivalent to model_distribution_network.ModelDistributionNetwork.

  We keep the implementations separate to minimize cluttering the implementation
  of the main method.
  """

  def __init__(self,
               observation_spec,
               action_spec,
               latent1_first_prior_distribution_ctor=ConstantMultivariateNormalDiag,
               latent1_prior_distribution_ctor=MultivariateNormalDiag,
               latent1_posterior_distribution_ctor=MultivariateNormalDiag,
               latent2_prior_distribution_ctor=MultivariateNormalDiag,
               latent2_posterior_distribution_ctor=MultivariateNormalDiag,
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               skip_first_kl=False,
               sequential_latent1_prior=True,
               sequential_latent2_prior=True,
               sequential_latent1_posterior=True,
               sequential_latent2_posterior=True,
               model_reward=False,
               model_discount=False,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               reward_stddev=None,
               name=None):
    super(SlacModelDistributionNetwork, self).__init__(name=name)
    self.observation_spec = observation_spec
    self.action_spec = action_spec
    self.base_depth = base_depth
    self.latent1_size = latent1_size
    self.latent2_size = latent2_size
    self.kl_analytic = kl_analytic
    self.skip_first_kl = skip_first_kl
    self.model_reward = model_reward
    self.model_discount = model_discount

    # p(z_1^1)
    self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
    # p(z_1^2 | z_1^1)
    self.latent2_first_prior = latent2_prior_distribution_ctor(8 * base_depth, latent2_size)
    if sequential_latent1_prior:
      # p(z_{t+1}^1 | z_t^2, a_t)
      self.latent1_prior = latent1_prior_distribution_ctor(8 * base_depth, latent1_size)
    else:
      # p(z_{t+1}^1)
      self.latent1_prior = lambda prev_latent, prev_action: self.latent1_first_prior(prev_latent[..., 0])  # prev_latent is only used to determine the batch shape
    if sequential_latent2_prior:
      # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
      self.latent2_prior = latent2_prior_distribution_ctor(8 * base_depth, latent2_size)
    else:
      # p(z_{t+1}^2 | z_{t+1}^1)
      self.latent2_prior = lambda latent1, prev_latent2, prev_action: self.latent2_first_prior(latent1)

    # q(z_1^1 | x_1)
    self.latent1_first_posterior = latent1_posterior_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    if latent2_posterior_distribution_ctor == latent2_prior_distribution_ctor:
      self.latent2_first_posterior = self.latent2_first_prior  # share
    else:
      self.latent2_first_posterior = latent2_posterior_distribution_ctor(8 * base_depth, latent2_size)
    if sequential_latent1_posterior:
      # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
      self.latent1_posterior = latent1_posterior_distribution_ctor(8 * base_depth, latent1_size)
    else:
      # q(z_{t+1}^1 | x_{t+1})
      self.latent1_posterior = lambda feature, prev_latent2, prev_action: self.latent1_first_posterior(feature)
    if sequential_latent2_posterior:
      # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
      if latent2_posterior_distribution_ctor == latent2_prior_distribution_ctor:
        self.latent2_posterior = self.latent2_prior
      else:
        self.latent2_posterior = latent2_posterior_distribution_ctor(8 * base_depth, latent2_size)
    else:
      # q(z_{t+1}^2 | z_{t+1}^1) = p(z_{t+1}^2 | z_{t+1}^1)
      self.latent2_posterior = lambda latent1, prev_latent2, prev_action: self.latent2_first_posterior(latent1)

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
        latent2_posterior_samples[:, :sequence_length],
        actions[:, :sequence_length])
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

    if self.kl_analytic:
      latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
    else:
      latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                - latent1_prior_dists.log_prob(latent1_posterior_samples))
    if self.skip_first_kl:
      latent1_kl_divergences = latent1_kl_divergences[:, 1:]
    latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
    outputs.update({
        'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),
    })
    if self.latent2_posterior == self.latent2_prior:
      latent2_kl_divergences = 0.0
    else:
      if self.kl_analytic:
        latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
      else:
        latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)
                                  - latent2_prior_dists.log_prob(latent2_posterior_samples))
      if self.skip_first_kl:
        latent2_kl_divergences = latent2_kl_divergences[:, 1:]
      latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)
      outputs.update({
          'latent2_kl_divergence': tf.reduce_mean(latent2_kl_divergences),
      })
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

    try:
      latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    except:
      latent1_dists = None
    latent1_samples = tf.stack(latent1_samples, axis=1)
    try:
      latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    except:
      latent2_dists = None
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
        prev_latent2_sample = latent2_samples[t-1]
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        latent1_first_dist = self.latent1_first_posterior(features[t])
        latent1_dist = self.latent1_posterior(features[t], prev_latent2_sample, actions[t-1])
        latent1_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
        latent1_sample = latent1_dist.sample()
        latent2_first_dist = self.latent2_first_posterior(latent1_sample)
        latent2_dist = self.latent2_posterior(latent1_sample, prev_latent2_sample, actions[t-1])
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


@gin.configurable
class SimpleModelDistributionNetwork(tf.Module):

  def __init__(self,
               observation_spec,
               action_spec,
               base_depth=32,
               latent_size=256,
               kl_analytic=True,
               sequential_latent_prior=True,
               sequential_latent_posterior=True,
               model_reward=False,
               model_discount=False,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               reward_stddev=None,
               name=None):
    super(SimpleModelDistributionNetwork, self).__init__(name=name)
    self.observation_spec = observation_spec
    self.action_spec = action_spec
    self.base_depth = base_depth
    self.latent_size = latent_size
    self.kl_analytic = kl_analytic
    self.model_reward = model_reward
    self.model_discount = model_discount

    # p(z_1)
    self.latent_first_prior = ConstantMultivariateNormalDiag(latent_size)
    if sequential_latent_prior:
      # p(z_{t+1} | z_t, a_t)
      self.latent_prior = MultivariateNormalDiag(8 * base_depth, latent_size)
    else:
      # p(z_{t+1})
      self.latent_prior = lambda prev_latent, prev_action: self.latent_first_prior(prev_latent[..., 0])  # prev_latent is only used to determine the batch shape

    # q(z_1 | x_1)
    self.latent_first_posterior = MultivariateNormalDiag(8 * base_depth, latent_size)
    if sequential_latent_posterior:
      # q(z_{t+1} | x_{t+1}, z_t, a_t)
      self.latent_posterior = MultivariateNormalDiag(8 * base_depth, latent_size)
    else:
      # q(z_{t+1} | x_{t+1})
      self.latent_posterior = lambda feature, prev_latent, prev_action: self.latent_first_posterior(feature)

    # compresses x_t into a vector
    self.compressor = Compressor(base_depth, 8 * base_depth)
    # p(x_t | z_t)
    self.decoder = Decoder(base_depth, scale=decoder_stddev)

    if self.model_reward:
      # p(r_t | z_t, a_t, z_{t+1})
      self.reward_predictor = Normal(8 * base_depth, scale=reward_stddev)
    else:
      self.reward_predictor = None
    if self.model_discount:
      # p(d_t | z_{t+1})
      self.discount_predictor = Bernoulli(8 * base_depth)
    else:
      self.discount_predictor = None

  @property
  def state_size(self):
    return self.latent_size

  def compute_loss(self, images, actions, step_types, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
    sequence_length = step_types.shape[1].value - 1

    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    latent_posterior_samples, latent_posterior_dists = latent_posterior_samples_and_dists
    latent_prior_samples, _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization
    latent_conditional_prior_samples, _ = self.sample_prior_or_posterior(
        actions, step_types, images=images[:, :1])  # for visualization. condition on first image only

    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent_size])
    latent_first_prior_dists = self.latent_first_prior(step_types)
    # these distributions start at t=1 and the inputs are from t-1
    latent_after_first_prior_dists = self.latent_prior(
        latent_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent_reset_masks),
        latent_first_prior_dists,
        latent_after_first_prior_dists)

    outputs = {}

    if self.kl_analytic:
      latent_kl_divergences = tfd.kl_divergence(latent_posterior_dists, latent_prior_dists)
    else:
      latent_kl_divergences = (latent_posterior_dists.log_prob(latent_posterior_samples)
                               - latent_prior_dists.log_prob(latent_posterior_samples))
    latent_kl_divergences = tf.reduce_sum(latent_kl_divergences, axis=1)
    outputs.update({
        'latent_kl_divergence': tf.reduce_mean(latent_kl_divergences),
    })
    outputs.update({
        'kl_divergence': tf.reduce_mean(latent_kl_divergences),
    })

    likelihood_dists = self.decoder(latent_posterior_samples)
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
    elbo = likelihood_log_probs - latent_kl_divergences

    if self.model_reward:
      reward_dists = self.reward_predictor(
          latent_posterior_samples[:, :sequence_length],
          actions[:, :sequence_length],
          latent_posterior_samples[:, 1:sequence_length + 1])
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
          latent_posterior_samples[:, 1:sequence_length + 1])
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
    prior_images = self.decoder(latent_prior_samples).mean()
    conditional_prior_images = self.decoder(latent_conditional_prior_samples).mean()

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

    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      is_conditional = images is not None and (t < images.shape[1].value)
      if t == 0:
        if is_conditional:
          latent_dist = self.latent_first_posterior(features[t])
        else:
          latent_dist = self.latent_first_prior(step_types[t])  # step_types is only used to infer batch_size
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        if is_conditional:
          latent_first_dist = self.latent_first_posterior(features[t])
          latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        else:
          latent_first_dist = self.latent_first_prior(step_types[t])
          latent_dist = self.latent_prior(latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists

  def sample_posterior(self, images, actions, step_types, features=None):
    sequence_length = step_types.shape[1].value - 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.compressor(images)

    # swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])

    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent_dist = self.latent_first_posterior(features[t])
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        latent_first_dist = self.latent_first_posterior(features[t])
        latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists
