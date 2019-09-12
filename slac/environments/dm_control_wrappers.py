from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
import tensorflow as tf


class PixelObservationsDmControlWrapper(wrappers.PyEnvironmentBaseWrapper):

  def __init__(self, env, observations_whitelist=None, render_kwargs=None):
    super(PixelObservationsDmControlWrapper, self).__init__(env)
    if observations_whitelist is None:
      self._observations_whitelist = ['state', 'pixels']
    else:
      self._observations_whitelist = observations_whitelist
    self._render_kwargs = dict(
        width=64,
        height=64,
        camera_id=0,
    )
    if render_kwargs is not None:
      self._render_kwargs.update(render_kwargs)

    observation_spec = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observation_spec['state'] = self._env.observation_spec()
      elif observation_name == 'pixels':
        image_shape = (
            self._render_kwargs['height'], self._render_kwargs['width'], 3)
        image_spec = array_spec.BoundedArraySpec(
            shape=image_shape, dtype=np.uint8, minimum=0, maximum=255)
        observation_spec['pixels'] = image_spec
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    self._observation_spec = observation_spec

  def observation_spec(self):
    return self._observation_spec

  def _modify_observation(self, observation):
    observations = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observations['state'] = observation
      elif observation_name == 'pixels':
        def get_physics(env):
          if hasattr(env, 'physics'):
            return env.physics
          else:
            return get_physics(env.wrapped_env())
        image = get_physics(self._env).render(**self._render_kwargs)
        observations['pixels'] = image
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    return observations

  def _step(self, action):
    time_step = self._env.step(action)
    time_step = time_step._replace(
        observation=self._modify_observation(time_step.observation))
    return time_step

  def _reset(self):
    time_step = self._env.reset()
    time_step = time_step._replace(
        observation=self._modify_observation(time_step.observation))
    return time_step

  def render(self, mode='rgb_array'):
    return self._env.render(mode=mode)
