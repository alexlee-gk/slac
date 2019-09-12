from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from slac.utils import common as slac_common
from slac.utils import gif_utils
from slac.utils import nest_utils as slac_nest_utils

tfd = tfp.distributions


def _gif_summary(name, images, fps, saturate=False, step=None):
  images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
  output = tf.concat(tf.unstack(images), axis=2)[None]
  gif_utils.gif_summary_v2(name, output, 1, fps, step=step)


def _gif_and_image_summary(name, images, fps, saturate=False, step=None):
  images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
  output = tf.concat(tf.unstack(images), axis=2)[None]
  gif_utils.gif_summary_v2(name, output, 1, fps, step=step)
  output = tf.concat(tf.unstack(images), axis=2)
  output = tf.concat(tf.unstack(output), axis=0)[None]
  tf.contrib.summary.image(name, output, step=step)


def filter_before_first_step(time_steps, actions=None):
  flat_time_steps = tf.nest.flatten(time_steps)
  flat_time_steps = [tf.unstack(time_step, axis=1) for time_step in
                     flat_time_steps]
  time_steps = [tf.nest.pack_sequence_as(time_steps, time_step) for time_step in
                zip(*flat_time_steps)]
  if actions is None:
    actions = [None] * len(time_steps)
  else:
    actions = tf.unstack(actions, axis=1)
  assert len(time_steps) == len(actions)

  time_steps = list(reversed(time_steps))
  actions = list(reversed(actions))
  filtered_time_steps = []
  filtered_actions = []
  for t, (time_step, action) in enumerate(zip(time_steps, actions)):
    if t == 0:
      reset_mask = tf.equal(time_step.step_type, ts.StepType.FIRST)
    else:
      time_step = tf.nest.map_structure(lambda x, y: tf.where(reset_mask, x, y),
                                        last_time_step, time_step)
      action = tf.where(reset_mask, tf.zeros_like(action),
                        action) if action is not None else None
    filtered_time_steps.append(time_step)
    filtered_actions.append(action)
    reset_mask = tf.logical_or(
        reset_mask,
        tf.equal(time_step.step_type, ts.StepType.FIRST))
    last_time_step = time_step
  filtered_time_steps = list(reversed(filtered_time_steps))
  filtered_actions = list(reversed(filtered_actions))

  filtered_flat_time_steps = [tf.nest.flatten(time_step) for time_step in
                              filtered_time_steps]
  filtered_flat_time_steps = [tf.stack(time_step, axis=1) for time_step in
                              zip(*filtered_flat_time_steps)]
  filtered_time_steps = tf.nest.pack_sequence_as(filtered_time_steps[0],
                                                 filtered_flat_time_steps)
  if action is None:
    return filtered_time_steps
  else:
    actions = tf.stack(filtered_actions, axis=1)
    return filtered_time_steps, actions


class ActorSequencePolicy(tf_policy.Base):
  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               info_spec=(),
               actor_network=None,
               model_network=None,
               compressor_network=None,
               sequence_length=2,
               actor_input='state',
               control_timestep=None,
               num_images_per_summary=1,
               debug_summaries=False,
               name=None):
    if not isinstance(actor_network, network.Network):
      raise ValueError('actor_network must be a network.Network. Found '
                       '{}.'.format(type(actor_network)))
    self._actor_network = actor_network
    self._model_network = model_network
    self._compressor_network = compressor_network

    self._sequence_length = sequence_length
    self._actor_input = actor_input
    self._control_timestep = control_timestep
    self._num_images_per_summary = num_images_per_summary
    self._debug_summaries = debug_summaries

    def _add_time_dimension(spec):
      return tensor_spec.TensorSpec(
          (sequence_length,) + tuple(spec.shape), spec.dtype, spec.name)

    time_steps_spec = tf.nest.map_structure(
        _add_time_dimension, time_step_spec)
    actions_spec = tf.nest.map_structure(
        _add_time_dimension, action_spec)
    policy_state_spec = (
        actor_network.state_spec, time_steps_spec, actions_spec)

    super(ActorSequencePolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        name=name)

  def _apply_actor_network(self, time_steps, actions, network_state):
    states = []
    for actor_input in self._actor_input.split('__'):
      if actor_input == 'state':
        state = time_steps.observation['state'][:, -1]
      elif actor_input == 'latent':
        images = tf.image.convert_image_dtype(
            time_steps.observation['pixels'], tf.float32)
        latents, _ = self._model_network.sample_posterior(
            images, slac_common.flatten(actions, axis=2), time_steps.step_type)
        if isinstance(latents, (tuple, list)):
          latents = tf.concat(latents, axis=-1)
        state = latents[:, -1]
      elif actor_input == 'feature':
        image = tf.image.convert_image_dtype(
            time_steps.observation['pixels'][:, -1], tf.float32)
        state = self._compressor_network(image)
      elif actor_input in ('sequence_feature', 'sequence_action_feature'):
        filtered_time_steps, filtered_actions = filter_before_first_step(
            time_steps, actions)
        images = tf.image.convert_image_dtype(
            filtered_time_steps.observation['pixels'], tf.float32)
        features = self._compressor_network(images)
        sequence_feature = slac_common.flatten(features)
        if actor_input == 'sequence_action_feature':
          sequence_action = slac_common.flatten(filtered_actions[:, :-1])
          state = tf.concat([sequence_feature, sequence_action], axis=-1)
        else:
          state = sequence_feature
      else:
        raise NotImplementedError
      states.append(state)
    state = tf.concat(states, axis=-1)
    if self._debug_summaries:
      filtered_time_steps, filtered_actions = filter_before_first_step(
          time_steps, actions)
      images = tf.image.convert_image_dtype(
          filtered_time_steps.observation['pixels'], tf.float32)
      fps = 10 if self._control_timestep is None else int(
          np.round(1.0 / self._control_timestep))
      _gif_and_image_summary('ActorSequencePolicy/images',
                             images[:self._num_images_per_summary], fps,
                             step=self.train_step_counter)
    step_type = time_steps.step_type[:, -1]
    return self._actor_network(state, step_type, network_state)

  def _variables(self):
    variables = list(self._actor_network.variables)
    actor_inputs = set(self._actor_input.split('__'))
    if 'latent' in actor_inputs:
      variables += self._model_network.variables
    if {'feature', 'sequence_feature',
        'sequence_action_feature'} & actor_inputs:
      variables += self._compressor_network.variables
    return variables

  def _action(self, time_step, policy_state, seed):
    distribution_step = self.distribution(time_step, policy_state)
    action = distribution_step.action.sample(seed=seed)
    policy_state = distribution_step.state
    network_state, time_steps, actions = policy_state
    actions = tf.concat([actions[:, :-1], action[:, None]], axis=1)
    policy_state = network_state, time_steps, actions

    return distribution_step._replace(action=action, state=policy_state)

  def _distribution(self, time_step, policy_state):
    network_state, time_steps, actions = policy_state

    def _apply_sequence_update(tensors, tensor):
      return tf.concat([tensors, tensor[:, None]], axis=1)[:, 1:]

    time_steps = tf.nest.map_structure(
        _apply_sequence_update, time_steps, time_step)
    actions = tf.nest.map_structure(
        _apply_sequence_update, actions, tf.zeros_like(actions[:, 0]))

    # Actor network outputs nested structure of distributions or actions.
    action_or_distribution, network_state = self._apply_actor_network(
        time_steps, actions, network_state)

    policy_state = (network_state, time_steps, actions)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distribution = tf.nest.map_structure(_to_distribution,
                                         action_or_distribution)
    return policy_step.PolicyStep(distribution, policy_state)


@gin.configurable
class SlacAgent(tf_agent.TFAgent):
  """A SLAC Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               model_network,
               compressor_network,
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               model_optimizer,
               sequence_length,
               target_update_tau=1.0,
               target_update_period=1,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               initial_log_alpha=0.0,
               target_entropy=None,
               gradient_clipping=None,
               trainable_model=True,
               critic_input='state',
               actor_input='state',
               critic_input_stop_gradient=True,
               actor_input_stop_gradient=False,
               model_batch_size=None,
               control_timestep=None,
               num_images_per_summary=1,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):
    tf.Module.__init__(self, name=name)

    self._critic_network1 = critic_network
    self._critic_network2 = critic_network.copy(name='CriticNetwork2')
    self._target_critic_network1 = critic_network.copy(
        name='TargetCriticNetwork1')
    self._target_critic_network2 = critic_network.copy(
        name='TargetCriticNetwork2')
    self._actor_network = actor_network
    self._model_network = model_network
    self._compressor_network = compressor_network

    policy = ActorSequencePolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        model_network=self._model_network,
        compressor_network=self._compressor_network,
        sequence_length=sequence_length,
        actor_input=actor_input,
        control_timestep=control_timestep,
        num_images_per_summary=num_images_per_summary,
        debug_summaries=debug_summaries)

    self._log_alpha = common.create_variable(
        'initial_log_alpha',
        initial_value=initial_log_alpha,
        dtype=tf.float32,
        trainable=True)

    # If target_entropy was not passed, set it to negative of the total number
    # of action dimensions.
    if target_entropy is None:
      flat_action_spec = tf.nest.flatten(action_spec)
      target_entropy = -np.sum([
        np.product(single_spec.shape.as_list())
        for single_spec in flat_action_spec
      ])

    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self._alpha_optimizer = alpha_optimizer
    self._model_optimizer = model_optimizer
    self._sequence_length = sequence_length
    self._td_errors_loss_fn = td_errors_loss_fn
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_entropy = target_entropy
    self._gradient_clipping = gradient_clipping
    self._trainable_model = trainable_model
    self._critic_input = critic_input
    self._actor_input = actor_input
    self._critic_input_stop_gradient = critic_input_stop_gradient
    self._actor_input_stop_gradient = actor_input_stop_gradient
    self._model_batch_size = model_batch_size
    self._control_timestep = control_timestep
    self._num_images_per_summary = num_images_per_summary
    self._debug_summaries = debug_summaries
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._update_target = self._get_target_updater(
        tau=self._target_update_tau, period=self._target_update_period)

    self._actor_time_step_spec = time_step_spec._replace(
        observation=actor_network.input_tensor_spec)
    super(SlacAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=sequence_length + 1,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)
    self._train_model_fn = common.function_in_tf1()(self._train_model)

  def _initialize(self):
    """Returns an op to initialize the agent.

    Copies weights from the Q networks to the target Q network.
    """
    common.soft_variables_update(
        self._critic_network1.variables,
        self._target_critic_network1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._critic_network2.variables,
        self._target_critic_network2.variables,
        tau=1.0)

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)
    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    if (self.train_sequence_length is not None and
            self.train_sequence_length == 2):
      # Sequence empty time dimension if critic network is stateless.
      time_steps, actions, next_time_steps = tf.nest.map_structure(
          lambda t: tf.squeeze(t, axis=1),
          (time_steps, actions, next_time_steps))
    return time_steps, actions, next_time_steps

  def _train(self, experience, weights=None):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    time_steps, actions, next_time_steps = self._experience_to_transitions(
        experience)
    time_step, action, next_time_step = self._experience_to_transitions(
        tf.nest.map_structure(lambda x: x[:, -2:], experience))
    time_step, action, next_time_step = tf.nest.map_structure(
        lambda x: tf.squeeze(x, axis=1), (time_step, action, next_time_step))

    with tf.GradientTape(persistent=True) as tape:
      state_only = (self._actor_input == 'state' and
                    self._critic_input == 'state' and
                    not self._trainable_model)
      critic_and_actor_inputs = set(
          self._critic_input.split('__') + self._actor_input.split('__'))
      if not state_only:
        images = tf.image.convert_image_dtype(
            experience.observation['pixels'], tf.float32)
        features = self._compressor_network(images)
      if 'latent' in critic_and_actor_inputs:
        if self._compressor_network == self._model_network.compressor:
          latent_samples_and_dists = self._model_network.sample_posterior(
              images, actions, experience.step_type, features)
        else:
          latent_samples_and_dists = self._model_network.sample_posterior(
              images, actions, experience.step_type)
        latents, _ = latent_samples_and_dists
        if isinstance(latents, (tuple, list)):
          latents = tf.concat(latents, axis=-1)
        latent, next_latent = tf.unstack(latents[:, -2:], axis=1)
      else:
        latent_samples_and_dists = None
      if 'feature' in critic_and_actor_inputs:
        feature, next_feature = tf.unstack(features[:, -2:], axis=1)
      if {'sequence_feature',
          'sequence_action_feature'} & critic_and_actor_inputs:
        feature_time_steps = time_steps._replace(observation=features[:, :-1])
        next_feature_time_steps = next_time_steps._replace(
            observation=features[:, 1:])
        filtered_feature_time_steps, filtered_actions = (
            filter_before_first_step(feature_time_steps, actions))
        filtered_next_feature_time_steps = filter_before_first_step(
            next_feature_time_steps)
        sequence_feature = slac_common.flatten(
            filtered_feature_time_steps.observation)
        next_sequence_feature = slac_common.flatten(
            filtered_next_feature_time_steps.observation)
        if 'sequence_action_feature' in critic_and_actor_inputs:
          sequence_action = slac_common.flatten(filtered_actions[:, :-1])
          sequence_action_feature = tf.concat(
              [sequence_feature, sequence_action], axis=-1)
          next_sequence_action = slac_common.flatten(filtered_actions[:, 1:])
          next_sequence_action_feature = tf.concat(
              [next_sequence_feature, next_sequence_action], axis=-1)
      if self._debug_summaries:
        if not state_only:
          image_time_steps = time_steps._replace(observation=images[:, :-1])
          next_image_time_steps = next_time_steps._replace(
              observation=images[:, 1:])
          filtered_image_time_steps, _ = filter_before_first_step(
              image_time_steps, actions)
          filtered_next_image_time_steps = filter_before_first_step(
              next_image_time_steps)
          fps = 10 if self._control_timestep is None else int(
              np.round(1.0 / self._control_timestep))
          _gif_and_image_summary('images', filtered_image_time_steps.observation[
                                           :self._num_images_per_summary], fps,
                                 step=self.train_step_counter)
          _gif_and_image_summary('next_images',
                                 filtered_next_image_time_steps.observation[
                                 :self._num_images_per_summary], fps,
                                 step=self.train_step_counter)

      critic_states = []
      critic_next_states = []
      for critic_input in self._critic_input.split('__'):
        if critic_input == 'latent':
          critic_state = latent
          critic_next_state = next_latent
        elif critic_input == 'state':
          critic_state, critic_next_state = tf.unstack(
              experience.observation['state'][:, -2:], axis=1)
        elif critic_input == 'feature':
          critic_state = feature
          critic_next_state = next_feature
        elif critic_input == 'sequence_feature':
          critic_state = sequence_feature
          critic_next_state = next_sequence_feature
        elif critic_input == 'sequence_action_feature':
          critic_state = sequence_action_feature
          critic_next_state = next_sequence_action_feature
        else:
          raise NotImplementedError
        critic_states.append(critic_state)
        critic_next_states.append(critic_next_state)
      critic_state = tf.concat(critic_states, axis=-1)
      critic_next_state = tf.concat(critic_next_states, axis=-1)
      critic_time_step = time_step._replace(observation=critic_state)
      critic_next_time_step = next_time_step._replace(
          observation=critic_next_state)

      actor_states = []
      actor_next_states = []
      for actor_input in self._actor_input.split('__'):
        if actor_input == 'latent':
          actor_state = latent
          actor_next_state = next_latent
        elif actor_input == 'state':
          actor_state, actor_next_state = tf.unstack(
              experience.observation['state'][:, -2:], axis=1)
        elif actor_input == 'feature':
          actor_state = feature
          actor_next_state = next_feature
        elif actor_input == 'sequence_feature':
          actor_state = sequence_feature
          actor_next_state = next_sequence_feature
        elif actor_input == 'sequence_action_feature':
          actor_state = sequence_action_feature
          actor_next_state = next_sequence_action_feature
        else:
          raise NotImplementedError
        actor_states.append(actor_state)
        actor_next_states.append(actor_next_state)
      actor_state = tf.concat(actor_states, axis=-1)
      actor_next_state = tf.concat(actor_next_states, axis=-1)
      actor_time_step = time_step._replace(observation=actor_state)
      actor_next_time_step = next_time_step._replace(observation=actor_next_state)

      critic_loss = self.critic_loss(
          critic_time_step,
          action,
          critic_next_time_step,
          actor_next_time_step,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights)

      actor_loss = self.actor_loss(
          critic_time_step, actor_time_step, weights=weights)

      alpha_loss = self.alpha_loss(actor_time_step, weights=weights)

      if self._trainable_model:
        model_loss = self.model_loss(
            images,
            experience.action,
            experience.step_type,
            experience.reward,
            experience.discount,
            latent_posterior_samples_and_dists=latent_samples_and_dists,
            weights=weights)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_variables = (
        list(self._critic_network1.variables) +
        list(self._critic_network2.variables) +
        list(self._compressor_network.variables) +
        list(self._model_network.variables))
    assert critic_variables, 'No critic variables to optimize.'
    critic_grads = tape.gradient(critic_loss, critic_variables)
    self._apply_gradients(
        critic_grads, critic_variables, self._critic_optimizer)

    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_variables = (
        list(self._actor_network.variables) +
        list(self._compressor_network.variables) +
        list(self._model_network.variables))
    assert actor_variables, 'No actor variables to optimize.'
    actor_grads = tape.gradient(actor_loss, actor_variables)
    self._apply_gradients(actor_grads, actor_variables, self._actor_optimizer)

    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_variables = [self._log_alpha]
    assert alpha_variables, 'No alpha variable to optimize.'
    alpha_grads = tape.gradient(alpha_loss, alpha_variables)
    self._apply_gradients(alpha_grads, alpha_variables, self._alpha_optimizer)

    if self._trainable_model:
      tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
      model_variables = list(self._model_network.variables)
      assert model_variables, 'No model variables to optimize.'
      model_grads = tape.gradient(model_loss, model_variables)
      self._apply_gradients(model_grads, model_variables, self._model_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)
      if self._trainable_model:
        tf.compat.v2.summary.scalar(
            name='model_loss', data=model_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss
    if self._trainable_model:
      total_loss += model_loss

    return tf_agent.LossInfo(loss=total_loss, extra=())

  def _apply_gradients(self, gradients, variables, optimizer):
    grads_and_vars = zip(gradients, variables)
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target'):
      def update():
        """Update target network."""
        critic_update_1 = common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables, tau)
        critic_update_2 = common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables, tau)
        return tf.group(critic_update_1, critic_update_2)

      return common.Periodically(update, period, 'update_targets')

  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  actor_next_time_steps,
                  td_errors_loss_fn,
                  gamma=1.0,
                  reward_scale_factor=1.0,
                  weights=None):
    """Computes the critic loss for SAC training.

    Args:
      time_steps: A batch of timesteps for the critic.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps for the critic.
      actor_next_time_steps: A batch of next timesteps for the actor.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      if self._critic_input_stop_gradient:
        time_steps = tf.nest.map_structure(tf.stop_gradient, time_steps)
        next_time_steps = tf.nest.map_structure(tf.stop_gradient,
                                                next_time_steps)

      # not really necessary since there is a stop_gradient for the td_targets
      actor_next_time_steps = tf.nest.map_structure(tf.stop_gradient,
                                                    actor_next_time_steps)

      next_actions_distribution, _ = self._actor_network(
          actor_next_time_steps.observation, actor_next_time_steps.step_type)
      next_actions = next_actions_distribution.sample()
      next_log_pis = next_actions_distribution.log_prob(next_actions)
      target_input_1 = (next_time_steps.observation, next_actions)
      target_q_values1, unused_network_state1 = self._target_critic_network1(
          target_input_1, next_time_steps.step_type)
      target_input_2 = (next_time_steps.observation, next_actions)
      target_q_values2, unused_network_state2 = self._target_critic_network2(
          target_input_2, next_time_steps.step_type)
      target_q_values = (
          tf.minimum(target_q_values1, target_q_values2) -
          tf.exp(self._log_alpha) * next_log_pis)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_input_1 = (time_steps.observation, actions)
      pred_td_targets1, unused_network_state1 = self._critic_network1(
          pred_input_1, time_steps.step_type)
      pred_input_2 = (time_steps.observation, actions)
      pred_td_targets2, unused_network_state2 = self._critic_network2(
          pred_input_2, time_steps.step_type)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if weights is not None:
        critic_loss *= weights

      # Take the mean across the batch.
      critic_loss = tf.reduce_mean(input_tensor=critic_loss)

      if self._debug_summaries:
        td_errors1 = td_targets - pred_td_targets1
        td_errors2 = td_targets - pred_td_targets2
        td_errors = tf.concat([td_errors1, td_errors2], axis=0)
        common.generate_tensor_summaries('td_errors', td_errors,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_targets', td_targets,
                                         self.train_step_counter)
        common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                         self.train_step_counter)
        common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                         self.train_step_counter)

      return critic_loss

  def actor_loss(self, time_steps, actor_time_steps, weights=None):
    """Computes the actor_loss for SAC training.

    Args:
      time_steps: A batch of timesteps for the critic.
      actor_time_steps: A batch of timesteps for the actor.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      time_steps = tf.nest.map_structure(tf.stop_gradient, time_steps)

      if self._actor_input_stop_gradient:
        actor_time_steps = tf.nest.map_structure(tf.stop_gradient,
                                                 actor_time_steps)

      actions_distribution, _ = self._actor_network(
          actor_time_steps.observation, actor_time_steps.step_type)
      actions = actions_distribution.sample()
      log_pis = actions_distribution.log_prob(actions)
      target_input_1 = (time_steps.observation, actions)
      target_q_values1, unused_network_state1 = self._critic_network1(
          target_input_1, time_steps.step_type)
      target_input_2 = (time_steps.observation, actions)
      target_q_values2, unused_network_state2 = self._critic_network2(
          target_input_2, time_steps.step_type)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)
      actor_loss = tf.exp(self._log_alpha) * log_pis - target_q_values
      if weights is not None:
        actor_loss *= weights
      actor_loss = tf.reduce_mean(input_tensor=actor_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('actor_loss', actor_loss,
                                         self.train_step_counter)
        common.generate_tensor_summaries('actions', actions,
                                         self.train_step_counter)
        common.generate_tensor_summaries('log_pis', log_pis,
                                         self.train_step_counter)
        tf.compat.v2.summary.scalar(
            name='entropy_avg',
            data=-tf.reduce_mean(input_tensor=log_pis),
            step=self.train_step_counter)
        common.generate_tensor_summaries('target_q_values', target_q_values,
                                         self.train_step_counter)
        batch_size = nest_utils.get_outer_shape(
            time_steps, self._time_step_spec)[0]
        policy_state = self.policy.get_initial_state(batch_size)
        action_distribution = self.policy.distribution(
            time_steps, policy_state).action
        if isinstance(action_distribution, tfp.distributions.Normal):
          common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                           self.train_step_counter)
          common.generate_tensor_summaries(
              'act_stddev', action_distribution.scale, self.train_step_counter)
        elif isinstance(action_distribution, tfp.distributions.Categorical):
          common.generate_tensor_summaries(
              'act_mode', action_distribution.mode(), self.train_step_counter)
        try:
          common.generate_tensor_summaries('entropy_action',
                                           action_distribution.entropy(),
                                           self.train_step_counter)
        except NotImplementedError:
          pass  # Some distributions do not have an analytic entropy.

      return actor_loss

  def alpha_loss(self, actor_time_steps, weights=None):
    """Computes the alpha_loss for EC-SAC training.

    Args:
      actor_time_steps: A batch of timesteps for the actor.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      alpha_loss: A scalar alpha loss.
    """
    with tf.name_scope('alpha_loss'):
      actions_distribution, _ = self._actor_network(
          actor_time_steps.observation, actor_time_steps.step_type)
      actions = actions_distribution.sample()
      log_pis = actions_distribution.log_prob(actions)
      alpha_loss = (
          self._log_alpha * tf.stop_gradient(-log_pis - self._target_entropy))

      if weights is not None:
        alpha_loss *= weights

      alpha_loss = tf.reduce_mean(input_tensor=alpha_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                         self.train_step_counter)

      return alpha_loss

  def train_model(self, experience, weights=None):
    if self._enable_functions and getattr(
        self, "_train_model_fn", None) is None:
      raise RuntimeError(
          "Cannot find _train_model_fn.  Did %s.__init__ call super?"
          % type(self).__name__)
    if not isinstance(experience, trajectory.Trajectory):
      raise ValueError(
          "experience must be type Trajectory, saw type: %s" % type(experience))

    if self._enable_functions:
      loss_info = self._train_model_fn(experience=experience, weights=weights)
    else:
      loss_info = self._train_model(experience=experience, weights=weights)

    if not isinstance(loss_info, tf_agent.LossInfo):
      raise TypeError(
          "loss_info is not a subclass of LossInfo: {}".format(loss_info))
    return loss_info

  def _train_model(self, experience, weights=None):
    with tf.GradientTape() as tape:
      images = tf.image.convert_image_dtype(
          experience.observation['pixels'], tf.float32)
      model_loss = self.model_loss(
          images,
          experience.action,
          experience.step_type,
          rewards=experience.reward,
          discounts=experience.discount,
          weights=weights)
    tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
    model_variables = list(self._model_network.variables)
    assert model_variables, 'No model variables to optimize.'
    model_grads = tape.gradient(model_loss, model_variables)
    self._apply_gradients(model_grads, model_variables, self._model_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='model_loss', data=model_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)

    total_loss = model_loss

    return tf_agent.LossInfo(loss=total_loss, extra=())

  def model_loss(self,
                 images,
                 actions,
                 step_types,
                 rewards,
                 discounts,
                 latent_posterior_samples_and_dists=None,
                 weights=None):
    with tf.name_scope('model_loss'):
      if self._model_batch_size is not None:
        # Allow model batch size to be smaller than the batch size of the
        # other losses. This is because the model loss already gets a lot of
        # supervision from having a loss over all time steps.
        images, actions, step_types, rewards, discounts = tf.nest.map_structure(
            lambda x: x[:self._model_batch_size],
            (images, actions, step_types, rewards, discounts))
        if latent_posterior_samples_and_dists is not None:
          latent_posterior_samples, latent_posterior_dists = latent_posterior_samples_and_dists
          latent_posterior_samples = tf.nest.map_structure(
              lambda x: x[:self._model_batch_size], latent_posterior_samples)
          latent_posterior_dists = slac_nest_utils.map_distribution_structure(
              lambda x: x[:self._model_batch_size], latent_posterior_dists)
          latent_posterior_samples_and_dists = (
              latent_posterior_samples, latent_posterior_dists)

      model_loss, outputs = self._model_network.compute_loss(
          images, actions, step_types,
          rewards=rewards,
          discounts=discounts,
          latent_posterior_samples_and_dists=latent_posterior_samples_and_dists)
      for name, output in outputs.items():
        if output.shape.ndims == 0:
          tf.contrib.summary.scalar(name, output)
        elif output.shape.ndims == 5:
          fps = 10 if self._control_timestep is None else int(
              np.round(1.0 / self._control_timestep))
          if self._debug_summaries:
            _gif_summary(name + '/original',
                         output[:self._num_images_per_summary], fps,
                         step=self.train_step_counter)
          _gif_summary(name, output[:self._num_images_per_summary], fps,
                       saturate=True, step=self.train_step_counter)
        else:
          raise NotImplementedError

      if weights is not None:
        model_loss *= weights

      model_loss = tf.reduce_mean(input_tensor=model_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('model_loss', model_loss,
                                         self.train_step_counter)

      return model_loss
