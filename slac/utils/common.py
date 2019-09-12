from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def flatten(input, axis=1, end_axis=-1):
  """
  Caffe-style flatten.

  Args:
    inputs: An N-D tensor.
    axis: The first axis to flatten: all preceding axes are retained in the
      output. May be negative to index from the end (e.g., -1 for the last
      axis).
    end_axis: The last axis to flatten: all following axes are retained in the
      output. May be negative to index from the end (e.g., the default -1 for
      the last axis)
  Returns:
      A M-D tensor where M = N - (end_axis - axis)
  """
  input_shape = tf.shape(input)
  input_rank = tf.shape(input_shape)[0]
  if axis < 0:
    axis = input_rank + axis
  if end_axis < 0:
    end_axis = input_rank + end_axis
  output_shape = []
  if axis != 0:
    output_shape.append(input_shape[:axis])
  output_shape.append([tf.reduce_prod(input_shape[axis:end_axis + 1])])
  if end_axis + 1 != input_rank:
    output_shape.append(input_shape[end_axis + 1:])
  output_shape = tf.concat(output_shape, axis=0)
  output = tf.reshape(input, output_shape)
  return output
