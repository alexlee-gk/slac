from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.environments import wrappers


class VideoWrapper(wrappers.PyEnvironmentBaseWrapper):
  def __init__(self, env):
    """Wrapper that keeps track of frames that are rendered after every step.

    It is useful when this environment is wrapped by an ActionRepeat wrapper,
    in which the latter would not be able to render the frames in between steps.
    The `frames` buffer is unbounded and it should be periodically emptied
    elsewhere outside of this class.
    """
    super(VideoWrapper, self).__init__(env)
    self._frames = []
    self._rendering = False

  def _reset(self):
    time_step = self._env.reset()
    if self._rendering:
      self._frames.append(self._env.render())
    return time_step

  def _step(self, action):
    time_step = self._env.step(action)
    if self._rendering:
      self._frames.append(self._env.render())
    return time_step

  @property
  def frames(self):
    return self._frames

  @property
  def rendering(self):
    return self._rendering

  def start_rendering(self):
    self._rendering = True

  def stop_rendering(self):
    self._rendering = False
