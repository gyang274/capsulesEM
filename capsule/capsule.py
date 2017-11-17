"""An implementation of matrix capsule with EM routing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# ------------------------------------------------------------------------------#
# ------------------------------------ init ------------------------------------#
# ------------------------------------------------------------------------------#

def _get_variable_wrapper(
  name, shape=None, dtype=None, initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None
):
  """Wrapper over tf.get_variable().
  """

  with tf.device('/cpu:0'):
    var = tf.get_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )
  return var


def _get_weights_wrapper(name, shape, weights_decay_factor=None):
  """Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """
  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=tf.float32,
    initializer=tf.truncated_normal_initializer(
      mean=0.0, stddev=0.05, dtype=tf.float32
    )
  )

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'
    )

    tf.add_to_collection('losses', weights_wd)

  return weights


def _get_biases_wrapper(name, shape):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=tf.float32,
    initializer=tf.constant_initializer(0.0)
  )

  return biases

# ------------------------------------------------------------------------------#
# ------------------------------------ main ------------------------------------#
# ------------------------------------------------------------------------------#

def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, name):
  """Wrapper over tf.nn.conv2d().
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, name):
  """Wrapper over tf.nn.separable_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0
    )
    pkernel = _get_weights_wrapper(
      name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0
    )
    output = tf.nn.separable_conv2d(
      input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[pointwise_shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, name):
  """Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.depthwise_conv2d(
      inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(
        name='biases', shape=[d_]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )

  return output

# ------------------------------------------------------------------------------#
# ---------------------------------- division ----------------------------------#
# ------------------------------------------------------------------------------#

def _capsule_init(inputs, shape, strides, padding, pose_shape, name):
  """This constructs a primary capsule layer from a regular convolutional layer.

  :param inputs: a regular convoluation layer with shape [N, H, W, C],
    where often N is batch_size, H is height, W is width, and C is channel.
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: TODO
  :param padding: TODO
  :param pose_shape: the pose shape, [PH, PW], where PH is pose height, and PW is pose width.
  :param name: TODO
  :return: (pose, activation), pose: [N, H, W, C, PH, PW], activation: [N, H, W, C],
    where often N is batch_size, H is height, W is width, C is channel, PH is pose height, and PW is pose width.

  note: in paper, Figure1, this function provides the operation to build from
    ReLU Conv1 [batch_size, 14, 14, 32] to
    PrimaryCapsule [batch_size, 6, 6, 32, 4, 4] pose, [batch_size, 6, 6, 32] activation with
    Kernel [A, B, 4 x 4 + 1]
  """

  # assert len(pose_shape) == 2

  pose = []
  for ph in xrange(pose_shape[0]):
    pose_unit = []
    for pw in xrange(pose_shape[1]):
      pose_unit.append(
        _conv2d_wrapper(
          inputs, shape=shape, strides=strides, padding=padding, add_bias=False, name=name+'_pose_'+str(ph)+'_'+str(pw)
        )
      )
    pose.append(tf.stack(pose_unit, axis=-1, name=name+'_pose_'+str(ph)))
  pose = tf.stack(pose, axis=-1, name=name+'_pose')

  activation = _conv2d_wrapper(
    inputs, shape=shape, strides=strides, padding=padding, add_bias=False, name=name+'_actvation_conv'
  )

  activation = tf.sigmoid(
    activation, name=name+'_activation'
  )

  return pose, activation


def capsule_conv(inputs, shape, strides, name):
  """This constructs a convolution capsule layer from a primary or convolution capsule layer.

  :param inputs: a primary or convolution capsule layer with pose and activation,
    pose shape [N, H, W, C, PH, PW], activation shape [N, H, W, C]

  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: TODO
  :param name: TODO
  :return: (pose, activation) same as capsule_init

  TODO:
    at this moment, we assume shape I == inputs C, e.g., convolution is done across all inputs channels,
    and thus strides should have [1, SH, SW, 1].
  """

  inputs_pose, inputs_activation = inputs

  inputs_pose_shape = inputs_pose.get_shape().as_list()

  inputs_activation_shape = inputs_activation.get_shape().as_list()

  # TODO: allow shape I < inputs pose C and relax strides[3] == 1
  assert shape[2] == inputs_pose_shape[3]
  assert strides[0] == strides[-1] == 1

  # note: in paper, 1.1 Previous work on capsules:
  # 3. It uses a vector of length n rather than a matrix with n elements to represent a pose, so its
  # transformation matrices have n^2 parameters rather than just n.

  # this explicit express a matrix 4 x 4 should be use as a viewpoint transformation matrix to adjust pose.

  # TODO: vectorize the operation

  # figure out the number of scan
  with tf.variable_scope(name) as scope:
    for h_offset in xrange(0, inputs_pose_shape[1], strides[1]):
      for w_offset in xrange(0, inputs_pose_shape[2], strides[2]):
        vote_unit = []
        for output_channel in shape[-1]:
          # at each offset K x K x I x 4 x 4 - one output capsule in one output channel,
          # we need to stack O such capsule for one offset across all channels and do EM
          kernel = _get_weights_wrapper(
            name='pose_view_transform_weights'+'_'+str(h_offset)+'_'+str(w_offset)+'_'+str(output_channel),
            shape=[1] + shape[:-1] + inputs_pose_shape[-2:],
            weights_decay_factor=0.0
          )
          kernel = tf.tile(
            kernel, [inputs_pose_shape[0], 1, 1, 1, 1, 1], name='tile_kernel_match_batch_size'
          )
          # inputs_pose matmul with kernel (view transform) becomes vote
          vote_unit_item = tf.matmul(
            inputs_pose, kernel, name='pose_view_transform'
          )
          vote_unit.append(
            vote_unit_item
          )
        # vote_unit_shape: [N, KH, KW, I, O, PH, PW]
        vote_unit = tf.stack(
          vote_unit, axis=-3, name='one_offset_stack'
        )
        vote_unit_shape = vote_unit.get_shape().as_list()
        # reshape into [N, KH, KW, I, O, PH x PW]
        vote_unit = tf.reshape(
          vote_unit, vote_unit_shape[:-2] + [vote_unit_shape[-2] * vote_unit_shape[-1]]
        )
        # reshape into [N, KH x KW x I, O, PH x PW]
        vote_unit = tf.reshape(
          vote_unit, [vote_unit_shape[0], vote_unit_shape[1] * vote_unit_shape[2] * vote_unit_shape[3], vote_unit_shape[4], vote_unit_shape[5] * vote_unit_shape[6]]
        )
        # inputs_activation_unit_shape: [N, KH, KW, I]
        inputs_activation_unit = tf.slice(
          inputs_activation, [0, h_offset, w_offset, 0], [inputs_activation_shape[0], h_offset + shape[0] - 1, w_offset + shape[1] - 1, inputs_activation[-1]]
        )
        # reshape into [N, KH x KW x I]
        inputs_activation_unit = tf.reshape(
          inputs_activation_unit, [inputs_activation_shape[0], shape[0] * shape[1] * inputs_activation[-1]]
        )
        # EM
        pose_unit, activation_unit = matrix_capsule_em_routing(
          vote_unit, inputs_activation_unit
        )

  return pose, activation

def matrix_capsule_em_routing(vote, i_activation):

  # vote_shape: [N, KH x KW x I, O, PH x PW]
  vote_shape = vote.get_shape().as_list()
  # i_activation_shape: [N, KH x KW x I]
  i_activation_shape = i_activation.get_shape().as_list()

  rr = tf.constant(
    1.0/vote_shape[2], shape=vote_shape[:-1], dtype=tf.float32
  )

  return pose, activation












