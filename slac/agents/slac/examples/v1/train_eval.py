from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re
import time

import gin
import gin.tf
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_dm_control
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from slac.agents.slac import actor_distribution_network
from slac.agents.slac import critic_network
from slac.agents.slac import compressor_network
from slac.agents.slac import model_distribution_network
from slac.agents.slac import slac_agent
from slac.environments import dm_control_wrappers
from slac.environments import gym_wrappers
from slac.environments import video_wrapper
from slac.utils import gif_utils

nest = tf.contrib.framework.nest

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name used for naming the output directory.')
flags.DEFINE_string('train_eval_dir', None,
                    'Directory for writing the train and eval directories.'
                    'This flag is mutually exclusive to root_dir and '
                    'experiment_name.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


def get_train_eval_dir(root_dir, universe, env_name, domain_name, task_name,
                       experiment_name):
  root_dir = os.path.expanduser(root_dir)
  if universe == 'gym':
    train_eval_dir = os.path.join(root_dir, universe, env_name,
                                  experiment_name)
  elif universe == 'dm_control':
    train_eval_dir = os.path.join(root_dir, universe, domain_name, task_name,
                                  experiment_name)
  else:
    raise ValueError('Invalid universe %s.' % universe)
  return train_eval_dir


def load_environments(universe, env_name=None, domain_name=None, task_name=None,
                      render_size=128, observation_render_size=64,
                      observations_whitelist=None, action_repeat=1):
  """Loads train and eval environments.

  The universe can either be gym, in which case domain_name and task_name are
  ignored, or dm_control, in which case env_name is ignored.
  """
  if universe == 'gym':
    tf.compat.v1.logging.info(
        'Using environment {} from {} universe.'.format(env_name, universe))
    gym_env_wrappers = [
        functools.partial(gym_wrappers.RenderGymWrapper,
                          render_kwargs={'height': render_size,
                                         'width': render_size,
                                         'device_id': 0}),
        functools.partial(gym_wrappers.PixelObservationsGymWrapper,
                          observations_whitelist=observations_whitelist,
                          render_kwargs={'height': observation_render_size,
                                         'width': observation_render_size,
                                         'device_id': 0})]
    eval_gym_env_wrappers = [
        functools.partial(gym_wrappers.RenderGymWrapper,
                          render_kwargs={'height': render_size,
                                         'width': render_size,
                                         'device_id': 1}),
        # segfaults if the device is the same as train env
        functools.partial(gym_wrappers.PixelObservationsGymWrapper,
                          observations_whitelist=observations_whitelist,
                          render_kwargs={'height': observation_render_size,
                                         'width': observation_render_size,
                                         'device_id': 1})]  # segfaults if the device is the same as train env
    py_env = suite_mujoco.load(env_name, gym_env_wrappers=gym_env_wrappers)
    eval_py_env = suite_mujoco.load(env_name,
                                    gym_env_wrappers=eval_gym_env_wrappers)
  elif universe == 'dm_control':
    tf.compat.v1.logging.info(
        'Using domain {} and task {} from {} universe.'.format(domain_name,
                                                               task_name,
                                                               universe))
    render_kwargs = {
        'height': render_size,
        'width': render_size,
        'camera_id': 0,
    }
    dm_env_wrappers = [
        wrappers.FlattenObservationsWrapper,  # combine position and velocity
        functools.partial(dm_control_wrappers.PixelObservationsDmControlWrapper,
                          observations_whitelist=observations_whitelist,
                          render_kwargs={'height': observation_render_size,
                                         'width': observation_render_size,
                                         'camera_id': 0})]
    py_env = suite_dm_control.load(
        domain_name, task_name, render_kwargs=render_kwargs,
        env_wrappers=dm_env_wrappers)
    eval_py_env = suite_dm_control.load(
        domain_name, task_name, render_kwargs=render_kwargs,
        env_wrappers=dm_env_wrappers)
  else:
    raise ValueError('Invalid universe %s.' % universe)

  eval_py_env = video_wrapper.VideoWrapper(eval_py_env)

  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)
    eval_py_env = wrappers.ActionRepeat(eval_py_env, action_repeat)

  return py_env, eval_py_env


def compute_summaries(metrics,
                      environment,
                      policy,
                      num_episodes=1,
                      num_episodes_to_render=1,
                      images_ph=None,
                      images_summary=None,
                      render_images_summary=None):
  for metric in metrics:
    metric.reset()

  if num_episodes_to_render:
    environment.start_rendering()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  render_images = []
  if num_episodes_to_render and 'pixels' in time_step.observation:
    images = [[time_step.observation['pixels']]]
  else:
    images = []
  step = 0
  episode = 0
  while episode < num_episodes:
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)

    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    for observer in metrics:
      observer(traj)

    if episode < num_episodes_to_render:
      if traj.is_last():
        render_images.append(list(environment.frames))
        environment.frames[:] = []
        if episode + 1 >= num_episodes_to_render:
          environment.stop_rendering()

      if 'pixels' in time_step.observation:
        if traj.is_boundary():
          images.append([])
        images[-1].append(next_time_step.observation['pixels'])

    episode += np.sum(traj.is_last())
    step += np.sum(~traj.is_boundary())

    time_step = next_time_step
    policy_state = action_step.state

  py_metric.run_summaries(metrics)

  if render_images:
    render_images = pad_and_concatenate_videos(render_images)
    session = tf.compat.v1.get_default_session()
    session.run(render_images_summary, feed_dict={images_ph: [render_images]})
  if images:
    images = pad_and_concatenate_videos(images)
    session = tf.compat.v1.get_default_session()
    session.run(images_summary, feed_dict={images_ph: [images]})


def pad_and_concatenate_videos(videos):
  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  videos = [np.concatenate(frames, axis=1) for frames in zip(*videos)]
  return videos


def get_control_timestep(py_env):
  try:
    control_timestep = py_env.dt  # gym
  except AttributeError:
    control_timestep = py_env.control_timestep()  # dm_control
  return control_timestep


@gin.configurable
def train_eval(
    root_dir,
    experiment_name,
    train_eval_dir=None,
    universe='gym',
    env_name='HalfCheetah-v2',
    domain_name='cheetah',
    task_name='run',
    action_repeat=1,
    num_iterations=int(1e7),
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    model_network_ctor=model_distribution_network.ModelDistributionNetwork,
    critic_input='state',
    actor_input='state',
    compressor_descriptor='preprocessor_32_3',
    # Params for collect
    initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=int(1e5),
    # increase if necessary since buffers with images are huge
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    model_train_steps_per_iteration=1,
    initial_model_train_steps=100000,
    batch_size=256,
    model_batch_size=32,
    sequence_length=4,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    model_learning_rate=1e-4,
    td_errors_loss_fn=functools.partial(
        tf.compat.v1.losses.mean_squared_error, weights=0.5),
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for summaries and logging
    num_images_per_summary=1,
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=0, # enable if necessary since buffers with images are huge
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    gpu_memory_limit=None):
  """A simple train and eval for SLAC."""
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_memory_limit is None:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  else:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  if train_eval_dir is None:
    train_eval_dir = get_train_eval_dir(
        root_dir, universe, env_name, domain_name, task_name, experiment_name)
  train_dir = os.path.join(train_eval_dir, 'train')
  eval_dir = os.path.join(train_eval_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      py_metrics.AverageReturnMetric(
          name='AverageReturnEvalPolicy', buffer_size=num_eval_episodes),
      py_metrics.AverageEpisodeLengthMetric(
          name='AverageEpisodeLengthEvalPolicy',
          buffer_size=num_eval_episodes),
  ]
  eval_greedy_metrics = [
      py_metrics.AverageReturnMetric(
          name='AverageReturnEvalGreedyPolicy', buffer_size=num_eval_episodes),
      py_metrics.AverageEpisodeLengthMetric(
          name='AverageEpisodeLengthEvalGreedyPolicy',
          buffer_size=num_eval_episodes),
  ]
  eval_summary_flush_op = eval_summary_writer.flush()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # Create the environment.
    trainable_model = model_train_steps_per_iteration != 0
    state_only = (actor_input == 'state' and critic_input == 'state' and
                  not trainable_model and initial_model_train_steps == 0)
    # Save time from unnecessarily rendering observations.
    observations_whitelist = ['state'] if state_only else None
    py_env, eval_py_env = load_environments(
        universe, env_name=env_name, domain_name=domain_name,
        task_name=task_name,
        observations_whitelist=observations_whitelist,
        action_repeat=action_repeat)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=True)
    original_control_timestep = get_control_timestep(eval_py_env)
    control_timestep = original_control_timestep * float(action_repeat)
    fps = int(np.round(1.0 / control_timestep))
    render_fps = int(np.round(1.0 / original_control_timestep))

    # Get the data specs from the environment
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    if model_train_steps_per_iteration not in (0, train_steps_per_iteration):
      raise NotImplementedError
    model_net = model_network_ctor(observation_spec, action_spec)
    if compressor_descriptor == 'model':
      compressor_net = model_net.compressor
    elif re.match('preprocessor_(\d+)_(\d+)', compressor_descriptor):
      m = re.match('preprocessor_(\d+)_(\d+)', compressor_descriptor)
      filters, n_layers = m.groups()
      filters = int(filters)
      n_layers = int(n_layers)
      compressor_net = compressor_network.Preprocessor(
          filters, n_layers=n_layers)
    elif re.match('compressor_(\d+)', compressor_descriptor):
      m = re.match('compressor_(\d+)', compressor_descriptor)
      filters, = m.groups()
      filters = int(filters)
      compressor_net = compressor_network.Compressor(filters)
    elif re.match('softlearning_(\d+)_(\d+)', compressor_descriptor):
      m = re.match('softlearning_(\d+)_(\d+)', compressor_descriptor)
      filters, n_layers = m.groups()
      filters = int(filters)
      n_layers = int(n_layers)
      compressor_net = compressor_network.SoftlearningPreprocessor(
          filters, n_layers=n_layers)
    elif compressor_descriptor == 'd4pg':
      compressor_net = compressor_network.D4pgPreprocessor()
    else:
      raise NotImplementedError(compressor_descriptor)

    actor_state_size = 0
    for _actor_input in actor_input.split('__'):
      if _actor_input == 'state':
        state_size, = observation_spec['state'].shape
        actor_state_size += state_size
      elif _actor_input == 'latent':
        actor_state_size += model_net.state_size
      elif _actor_input == 'feature':
        actor_state_size += compressor_net.feature_size
      elif _actor_input in ('sequence_feature', 'sequence_action_feature'):
        actor_state_size += compressor_net.feature_size * sequence_length
        if _actor_input == 'sequence_action_feature':
          actor_state_size += tf.compat.dimension_value(
              action_spec.shape[0]) * (sequence_length - 1)
      else:
        raise NotImplementedError
    actor_input_spec = tensor_spec.TensorSpec((actor_state_size,),
                                              dtype=tf.float32)

    critic_state_size = 0
    for _critic_input in critic_input.split('__'):
      if _critic_input == 'state':
        state_size, = observation_spec['state'].shape
        critic_state_size += state_size
      elif _critic_input == 'latent':
        critic_state_size += model_net.state_size
      elif _critic_input == 'feature':
        critic_state_size += compressor_net.feature_size
      elif _critic_input in ('sequence_feature', 'sequence_action_feature'):
        critic_state_size += compressor_net.feature_size * sequence_length
        if _critic_input == 'sequence_action_feature':
          critic_state_size += tf.compat.dimension_value(
              action_spec.shape[0]) * (sequence_length - 1)
      else:
        raise NotImplementedError
    critic_input_spec = tensor_spec.TensorSpec((critic_state_size,),
                                               dtype=tf.float32)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        actor_input_spec,
        action_spec,
        fc_layer_params=actor_fc_layers)
    critic_net = critic_network.CriticNetwork(
        (critic_input_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers)

    tf_agent = slac_agent.SlacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        model_network=model_net,
        compressor_network=compressor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        model_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_learning_rate),
        sequence_length=sequence_length,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        trainable_model=trainable_model,
        critic_input=critic_input,
        actor_input=actor_input,
        model_batch_size=model_batch_size,
        control_timestep=control_timestep,
        num_images_per_summary=num_images_per_summary,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
    eval_greedy_py_policy = py_tf_policy.PyTFPolicy(
        greedy_policy.GreedyPolicy(tf_agent.policy))

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_py_metric.TFPyMetric(py_metrics.AverageReturnMetric(buffer_size=1)),
        tf_py_metric.TFPyMetric(
            py_metrics.AverageEpisodeLengthMetric(buffer_size=1)),
    ]

    collect_policy = tf_agent.collect_policy
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec, action_spec)

    initial_policy_state = initial_collect_policy.get_initial_state(1)
    initial_collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps).run(policy_state=initial_policy_state)

    policy_state = collect_policy.get_initial_state(1)
    collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration).run(policy_state=policy_state)

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
      return ~trajectories.is_boundary()[-2]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=sequence_length + 1).unbatch().filter(
        _filter_invalid_transition).batch(
        batch_size, drop_remainder=True).prefetch(3)
    dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    trajectories, unused_info = dataset_iterator.get_next()

    train_op = tf_agent.train(trajectories)
    summary_ops = []
    for train_metric in train_metrics:
      summary_ops.append(train_metric.tf_summaries(
          train_step=global_step, step_metrics=train_metrics[:2]))

    if initial_model_train_steps:
      with tf.name_scope('initial'):
        model_train_op = tf_agent.train_model(trajectories)
        model_summary_ops = []
        for summary_op in tf.compat.v1.summary.all_v2_summary_ops():
          if summary_op not in summary_ops:
            model_summary_ops.append(summary_op)

    with eval_summary_writer.as_default(), \
         tf.compat.v2.summary.record_if(True):
      for eval_metric in eval_metrics + eval_greedy_metrics:
        eval_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])
      if eval_interval:
        eval_images_ph = tf.compat.v1.placeholder(
            dtype=tf.uint8, shape=[None] * 5)
        eval_images_summary = gif_utils.gif_summary_v2(
            'ObservationVideoEvalPolicy', eval_images_ph, 1, fps)
        eval_render_images_summary = gif_utils.gif_summary_v2(
            'VideoEvalPolicy', eval_images_ph, 1, render_fps)
        eval_greedy_images_summary = gif_utils.gif_summary_v2(
            'ObservationVideoEvalGreedyPolicy', eval_images_ph, 1, fps)
        eval_greedy_render_images_summary = gif_utils.gif_summary_v2(
            'VideoEvalGreedyPolicy', eval_images_ph, 1, render_fps)

    train_config_saver = gin.tf.GinConfigSaverHook(
        train_dir, summarize_config=False)
    eval_config_saver = gin.tf.GinConfigSaverHook(
        eval_dir, summarize_config=False)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=2)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=tf_agent.policy,
        global_step=global_step,
        max_to_keep=2)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    with tf.compat.v1.Session() as sess:
      # Initialize graph.
      train_checkpointer.initialize_or_restore(sess)
      rb_checkpointer.initialize_or_restore(sess)

      # Initialize training.
      sess.run(dataset_iterator.initializer)
      common.initialize_uninitialized_variables(sess)
      sess.run(train_summary_writer.init())
      sess.run(eval_summary_writer.init())

      train_config_saver.after_create_session(sess)
      eval_config_saver.after_create_session(sess)

      global_step_val = sess.run(global_step)

      if global_step_val == 0:
        if eval_interval:
          # Initial eval of randomly initialized policy
          for _eval_metrics, _eval_py_policy, \
              _eval_render_images_summary, _eval_images_summary in (
              (eval_metrics, eval_py_policy,
               eval_render_images_summary, eval_images_summary),
              (eval_greedy_metrics, eval_greedy_py_policy,
               eval_greedy_render_images_summary, eval_greedy_images_summary)):
            compute_summaries(
                _eval_metrics,
                eval_py_env,
                _eval_py_policy,
                num_episodes=num_eval_episodes,
                num_episodes_to_render=num_images_per_summary,
                images_ph=eval_images_ph,
                render_images_summary=_eval_render_images_summary,
                images_summary=_eval_images_summary)
          sess.run(eval_summary_flush_op)

        # Run initial collect.
        logging.info('Global step %d: Running initial collect op.',
                     global_step_val)
        sess.run(initial_collect_op)

        # Checkpoint the initial replay buffer contents.
        rb_checkpointer.save(global_step=global_step_val)

        logging.info('Finished initial collect.')
      else:
        logging.info('Global step %d: Skipping initial collect op.',
                     global_step_val)

      policy_state_val = sess.run(policy_state)
      collect_call = sess.make_callable(collect_op, feed_list=[policy_state])
      train_step_call = sess.make_callable([train_op, summary_ops])
      if initial_model_train_steps:
        model_train_step_call = sess.make_callable(
            [model_train_op, model_summary_ops])
      global_step_call = sess.make_callable(global_step)

      timed_at_step = global_step_call()
      time_acc = 0
      steps_per_second_ph = tf.compat.v1.placeholder(
          tf.float32, shape=(), name='steps_per_sec_ph')
      # steps_per_second summary should always be recorded since it's only called every log_interval steps
      with tf.compat.v2.summary.record_if(True):
        steps_per_second_summary = tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_second_ph,
            step=global_step)

      for iteration in range(
            global_step_val, initial_model_train_steps + num_iterations):
        start_time = time.time()
        if iteration < initial_model_train_steps:
          total_loss_val, _ = model_train_step_call()
        else:
          time_step_val, policy_state_val = collect_call(policy_state_val)
          for _ in range(train_steps_per_iteration):
            total_loss_val, _ = train_step_call()

        time_acc += time.time() - start_time
        global_step_val = global_step_call()
        if log_interval and global_step_val % log_interval == 0:
          logging.info('step = %d, loss = %f',
                       global_step_val, total_loss_val.loss)
          steps_per_sec = (global_step_val - timed_at_step) / time_acc
          logging.info('%.3f steps/sec', steps_per_sec)
          sess.run(
              steps_per_second_summary,
              feed_dict={steps_per_second_ph: steps_per_sec})
          timed_at_step = global_step_val
          time_acc = 0

        if (train_checkpoint_interval and
            global_step_val % train_checkpoint_interval == 0):
          train_checkpointer.save(global_step=global_step_val)

        if iteration < initial_model_train_steps:
          continue

        if eval_interval and global_step_val % eval_interval == 0:
          for _eval_metrics, _eval_py_policy, \
              _eval_render_images_summary, _eval_images_summary in (
              (eval_metrics, eval_py_policy,
               eval_render_images_summary, eval_images_summary),
              (eval_greedy_metrics, eval_greedy_py_policy,
               eval_greedy_render_images_summary, eval_greedy_images_summary)):
            compute_summaries(
                _eval_metrics,
                eval_py_env,
                _eval_py_policy,
                num_episodes=num_eval_episodes,
                num_episodes_to_render=num_images_per_summary,
                images_ph=eval_images_ph,
                render_images_summary=_eval_render_images_summary,
                images_summary=_eval_images_summary)
          sess.run(eval_summary_flush_op)

        if (policy_checkpoint_interval and
            global_step_val % policy_checkpoint_interval == 0):
          policy_checkpointer.save(global_step=global_step_val)

        if (rb_checkpoint_interval and
            global_step_val % rb_checkpoint_interval == 0):
          rb_checkpointer.save(global_step=global_step_val)


def main(argv):
  tf.compat.v1.enable_resource_variables()
  FLAGS(argv)  # raises UnrecognizedFlagError for undefined flags
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param,
                                      skip_unknown=False)
  train_eval(FLAGS.root_dir, FLAGS.experiment_name,
             train_eval_dir=FLAGS.train_eval_dir)


if __name__ == '__main__':
  def validate_mutual_exclusion(flags_dict):
    valid_1 = (flags_dict['root_dir'] is not None and
               flags_dict['experiment_name'] is not None and
               flags_dict['train_eval_dir'] is None)
    valid_2 = (flags_dict['root_dir'] is None and
               flags_dict['experiment_name'] is None and
               flags_dict['train_eval_dir'] is not None)
    if valid_1 or valid_2:
      return True
    message = ('Exactly both root_dir and experiment_name or only '
               'train_eval_dir must be specified.')
    raise flags.ValidationError(message)

  flags.register_multi_flags_validator(
      ['root_dir', 'experiment_name', 'train_eval_dir'],
      validate_mutual_exclusion)
  app.run(main)
