"""An implementation of matrix capsules with EM routing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import pi

import tensorflow as tf

epsilon = 1e-9

# ------------------------------------------------------------------------------#
# ------------------------------------ init ------------------------------------#
# ------------------------------------------------------------------------------#

def _matmul_broadcast(x, y, name):
  """Compute x @ y, broadcasting over the first `N - 2` ranks
  """
  with tf.variable_scope(name) as scope:
    return tf.reduce_sum(
      x[..., tf.newaxis] * y[..., tf.newaxis, :, :], axis=-2
    )

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

def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
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
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
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
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
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
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output

# ------------------------------------------------------------------------------#
# ---------------------------------- capsules ----------------------------------#
# ------------------------------------------------------------------------------#

def capsule_init(inputs, shape, strides, padding, pose_shape, name):
  """This constructs a primary capsule layer from a regular convolution layer.

  :param inputs: a regular convolution layer with shape [N, H, W, C],
    where often N is batch_size, H is height, W is width, and C is channel.
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param padding: padding, often SAME or VALID.
  :param pose_shape: the shape of each pose matrix, [PH, PW],
    where PH is pose height, and PW is pose width.
  :param name: name.

  :return: (poses, activations),
    poses: [N, H, W, C, PH, PW], activations: [N, H, W, C],
    where often N is batch_size, H is output height, W is output width, C is output channels,
    and PH is pose height, and PW is pose width.

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    this function provides the operation to build from
    ReLU Conv1 [batch_size, 14, 14, A] to
    PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] with
    Kernel [A, B, 4 x 4 + 1], specifically,
    weight kernel shape [1, 1, A, B], strides [1, 1, 1, 1], pose_shape [4, 4]
  """

  # assert len(pose_shape) == 2

  with tf.variable_scope(name) as scope:

    # poses: build one by one
    # poses = []
    # for ph in xrange(pose_shape[0]):
    #   poses_wire = []
    #   for pw in xrange(pose_shape[1]):
    #     poses_unit = _conv2d_wrapper(
    #       inputs, shape=shape, strides=strides, padding=padding, add_bias=False, activation_fn=None, name=name+'_pose_'+str(ph)+'_'+str(pw)
    #     )
    #     poses_wire.append(poses_unit)
    #   poses.append(tf.stack(poses_wire, axis=-1, name=name+'_poses_'+str(ph)))
    # poses = tf.stack(poses, axis=-1, name=name+'_poses')

    # poses: simplified build all at once
    poses = _conv2d_wrapper(
      inputs,
      shape=shape[0:-1] + [shape[-1] * pose_shape[0] * pose_shape[1]],
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=None,
      name='pose_stacked'
    )
    poses_shape = poses.get_shape().as_list()
    poses = tf.reshape(
      poses, shape=poses_shape[0:-1] + [shape[-1], pose_shape[0], pose_shape[1]], name='pose'
    )

    activations = _conv2d_wrapper(
      inputs,
      shape=shape,
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=tf.sigmoid,
      name='activations'
    )

  return poses, activations


def capsule_conv(inputs, shape, strides, inverse_temperature, iterations, name):
  """This constructs a convolution capsule layer from a primary or convolution capsule layer.

  :param inputs: a primary or convolution capsule layer with poses and activations,
    poses shape [N, H, W, C, PH, PW], activations shape [N, H, W, C]
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param inverse_temperature: inverse temperature, \lambda,
    often determined by the a fix schedule w.r.t global_steps.
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (poses, activations) same as capsule_init().

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    this function provides the operation to build from
    PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] to
    ConvCapsule1 poses [batch_size, 6, 6, C, 4, 4], activations [batch_size, 6, 6, C] with
    Kernel [KH=3, KW=3, B, C, 4, 4], specifically,
    weight kernel shape [3, 3, B, C], strides [1, 2, 2, 1], pose_shape [4, 4]

    also, this function provides the operation to build from
    ConvCapsule1 poses [batch_size, 6, 6, C, 4, 4], activations [batch_size, 6, 6, C] to
    ConvCapsule2 poses [batch_size, 4, 4, D, 4, 4], activations [batch_size, 4, 4, D] with
    Kernel [KH=3, KW=3, C, D, 4, 4], specifically,
    weight kernel shape [3, 3, C, D], strides [1, 1, 1, 1], pose_shape [4, 4]
  """

  inputs_poses, inputs_activations = inputs

  inputs_poses_shape = inputs_poses.get_shape().as_list()

  inputs_activations_shape = inputs_activations.get_shape().as_list()

  assert shape[2] == inputs_poses_shape[3]
  assert strides[0] == strides[-1] == 1

  # note: with respect to the paper, matrix capsules with EM routing, 1.1 previous work on capsules:
  # 3. it uses a vector of length n rather than a matrix with n elements to represent a pose, so its transformation matrices have n^2 parameters rather than just n.

  # this explicit express a matrix PH x PW should be use as a viewpoint transformation matrix to adjust pose.

  with tf.variable_scope(name) as scope:

    # kernel: [KH, KW, I, O, PH, PW]
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights', shape=shape + inputs_poses_shape[-2:]
    )

    # note: https://github.com/tensorflow/tensorflow/issues/216
    # tf.matmul doesn't support for broadcasting at this moment, work around with _matmul_broadcast().
    # construct conv patches (this should be a c++ dedicated function support for capsule convolution)
    hk_offsets = [
      [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in
      range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
    ]
    wk_offsets = [
      [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in
      range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
    ]

    # inputs_poses [N, H, W, I, PH, PW] patches into [N, OH, OW, KH, KW, I, 1, PH, PW]
    # where OH, OW are output height and width determined by H, W, shape and strides,
    # and KH and KW are kernel height and width determined by shape
    inputs_poses_patches = tf.transpose(
      tf.gather(
        tf.gather(
          inputs_poses, hk_offsets, axis=1, name='gather_poses_height_kernel'
        ), wk_offsets, axis=3, name='gather_poses_width_kernel'
      ), perm=[0, 1, 3, 2, 4, 5, 6, 7], name='inputs_poses_patches'
    )
    # inputs_poses_patches expand dimensions from [N, OH, OW, KH, KW, I, PH, PW] to [N, OH, OW, KH, KW, I, 1, PH, PW]
    inputs_poses_patches = inputs_poses_patches[..., tf.newaxis, :, :]
    # inputs_votes: [N, OH, OW, KH, KW, I, O, PH, PW]
    # inputs_votes should be the inputs_poses_patches multiply with the kernel view transformation matrix
    votes = _matmul_broadcast(
      inputs_poses_patches, kernel, name='votes'
    )
    votes_shape = votes.get_shape().as_list()
    # inputs_votes: reshape into [N, OH, OW, KH x KW x I, O, PH x PW]
    votes = tf.reshape(
      votes, [
        votes_shape[0],  votes_shape[1],  votes_shape[2],
        votes_shape[3] * votes_shape[4] * votes_shape[5],
        votes_shape[6],  votes_shape[7] * votes_shape[8]
      ]
    )

    # inputs_activations: [N, H, W, I] patches into [N, OH, OW, KH, KW, I]
    inputs_activations_patches = tf.transpose(
      tf.gather(
        tf.gather(
          inputs_activations, hk_offsets, axis=1, name='gather_activations_height_kernel'
        ), wk_offsets, axis=3, name='gather_activations_width_kernel'
      ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
    )
    # inputs_activations: [N, OH, OW, KH, KW, I] reshape into [N, OH, OW, KH x KW x I]
    # re-use votes_shape so that make sure the votes and i_activations shape match each other.
    i_activations = tf.reshape(
      inputs_activations_patches, [
        votes_shape[0],  votes_shape[1],  votes_shape[2],
        votes_shape[3] * votes_shape[4] * votes_shape[5]
      ]
    )

    # beta_v and beta_a one for each output capsule: [N, OH, OW, O]
    beta_v = _get_weights_wrapper(
      name='beta_v', shape=[
        votes_shape[0], votes_shape[1], votes_shape[2], votes_shape[6]
      ]
    )
    beta_a = _get_weights_wrapper(
      name='beta_a', shape=[
        votes_shape[0], votes_shape[1], votes_shape[2], votes_shape[6]
      ]
    )

    # output poses and activations via matrix capsules_em_routing algorithm
    # this operation involves inputs and output capsules across all (hk_offsets, wk_offsets), across all channels
    # poses: [N, OH, OW, O, PH x PW], activations: [N, OH, OW, O]
    poses, activations = matrix_capsules_em_routing(
      votes, i_activations, beta_v, beta_a, inverse_temperature, iterations, name='em_routing'
    )
    # poses: [N, OH, OW, O, PH, PW]
    poses = tf.reshape(
      poses, [
        votes_shape[0], votes_shape[1], votes_shape[2], votes_shape[6], votes_shape[7], votes_shape[8]
      ]
    )

  return poses, activations


def capsule_fc(inputs, num_classes, inverse_temperature, iterations, name):
  """This constructs an output layer from a primary or convolution capsule layer via
    a full-connected operation with one view transformation kernel matrix shared across each channel.

  :param inputs: a primary or convolution capsule layer with poses and activations,
    poses shape [N, H, W, C, PH, PW], activations shape [N, H, W, C]
  :param num_classes: number of classes.
  :param inverse_temperature: inverse temperature, \lambda,
    often determined by the a fix schedule w.r.t global_steps.
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) same as capsule_init().

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    This is the D -> E in figure.
    This step includes two major sub-steps:
      1. Apply one view transform weight matrix PH x PW (4 x 4) to each input channel, this view transform matrix is
        shared across (height, width) locations. This is the reason the kernel labelled in D has 1 x 1, and the reason
        the number of variables of weights is D x E x 4 x 4.
      2. Re-struct the inputs vote from [N, H, W, I, PH, PW] into [N, H x W x I, PH x PW],
        add scaled coordinate on first two elements, EM routing an output [N, O, PH x PW],
        and reshape output [N, O, PH, PW].
    The difference between fully-connected layer and convolution layer, is that:
      1. The corresponding kernel size KH, KW in this fully-connected here is actually the whole H, W, instead of 1, 1.
      2. The view transformation matrix is shared within KH, KW (i.e., H, W) in this fully-connected layer,
        whereas in the convolution capsule layer, the view transformation can be different for each capsule
        in the KH, KW, but shared across different (height, width) locations.
  """

  inputs_poses, inputs_activations = inputs

  inputs_poses_shape = inputs_poses.get_shape().as_list()

  inputs_activations_shape = inputs_activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # kernel: [I, O, PH, PW]
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights', shape=[inputs_poses_shape[3], num_classes] + inputs_poses_shape[-2:]
    )

    # inputs_pose_expansion: [N, H, W, I, 1, PH, PW]
    # inputs_pose_expansion: expand inputs_pose dimension to match with kernel for broadcasting,
    # share the transformation matrices between different positions of the same capsule type,
    # share the transformation matrices as kernel (1, 1) broadcasting to inputs pose expansion (H, W)
    inputs_poses_expansion = inputs_poses[..., tf.newaxis, :, :]

    # votes: [N, H, W, I, O, PH, PW]
    votes = _matmul_broadcast(
      inputs_poses_expansion, kernel, name='votes'
    )
    votes_shape = votes.get_shape().as_list()
    # votes: reshape into [N, H, W, I, O, PH x PW]
    votes = tf.reshape(
      votes, votes_shape[:-2] + [votes_shape[-2] * votes_shape[-1]]
    )

    # add scaled coordinate (row, column) of the center of the receptive field of each capsule
    # to the first two elements of its vote
    H = inputs_poses_shape[1]
    W = inputs_poses_shape[2]

    coordinate_offset_hh = tf.reshape(
      (tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1]
    )
    coordinate_offset_h0 = tf.constant(
      0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32
    )
    coordinate_offset_h = tf.stack(
      [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for _ in xrange(14)], axis=-1
    )

    coordinate_offset_ww = tf.reshape(
      (tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1]
    )
    coordinate_offset_w0 = tf.constant(
      0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32
    )
    coordinate_offset_w = tf.stack(
      [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for _ in xrange(14)], axis=-1
    )

    votes = votes + coordinate_offset_h + coordinate_offset_w

    # votes: reshape into [N, H x W x I, O, PH x PW]
    votes = tf.reshape(
      votes, [
        votes_shape[0],
        votes_shape[1] * votes_shape[2] * votes_shape[3],
        votes_shape[4],  votes_shape[5] * votes_shape[6]
      ]
    )

    # inputs_activations: [N, H, W, I]
    # inputs_activations: reshape into [N, H x W x I]
    i_activations = tf.reshape(
      inputs_activations, [
        inputs_activations_shape[0],
        inputs_activations_shape[1] * inputs_activations_shape[2] * inputs_activations_shape[3]
      ]
    )

    # beta_v and beta_a one for each output capsule: [N, O]
    beta_v = _get_weights_wrapper(
      name='beta_v', shape=[
        inputs_poses_shape[0], num_classes
      ]
    )
    beta_a = _get_weights_wrapper(
      name='beta_a', shape=[
        inputs_poses_shape[0], num_classes
      ]
    )

    # output poses and activations via matrix capsules_em_routing algorithm
    # poses: [N, O, PH x PW], activations: [N, O]
    poses, activations = matrix_capsules_em_routing(
      votes, i_activations, beta_v, beta_a, inverse_temperature, iterations, name='em_routing'
    )

    # pose: [N, O, PH, PW]
    poses = tf.reshape(
      poses, [
        votes_shape[0], votes_shape[4], votes_shape[5], votes_shape[6]
      ]
    )

  return poses, activations


def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, inverse_temperature, iterations, name):
  """The EM routing between input capsules (i) and output capsules (o).

  :param votes: [N, OH, OW, KH x KW x I, O, PH x PW] from capsule_conv(),
    or [N, KH x KW x I, O, PH x PW] from capsule_fc()
  :param i_activation: [N, OH, OW, KH x KW x I, O] from capsule_conv(),
    or [N, KH x KW x I, O] from capsule_fc()
  :param beta_v: [N, OH, OW, O] from capsule_conv(),
    or [N, O] from capsule_fc()
  :param beta_a: [N, OH, OW, O] from capsule_conv(),
    or [N, O] from capsule_fc()
  :param inverse_temperature: [] from capsule_conv(),
    or [] from capsule_fc()
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) of output capsules.

  note: the comment assumes arguments from capsule_conv(), remove OH, OW if from capsule_fc(),
    the function make sure is applicable to both cases by using negative index in argument axis.
  """

  # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
  votes_shape = votes.get_shape().as_list()
  # i_activations: [N, OH, OW, KH x KW x I]
  i_activations_shape = i_activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # note: match rr shape, i_activations shape with votes shape for broadcasting in EM routing

    # rr: [N, OH, OW, KH x KW x I, O, 1],
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(
      1.0/votes_shape[-2], shape=votes_shape[:-1] + [1], dtype=tf.float32
    )
    rr = tf.Print(
      rr, [rr.shape, rr[0, ..., :, :, :]], 'rr', summarize=20
    )

    # i_activations: expand_dims to [N, OH, OW, KH x KW x I, 1, 1]
    i_activations = i_activations[..., tf.newaxis, tf.newaxis]
    i_activations = tf.Print(
      i_activations, [i_activations.shape, i_activations[0, ..., :, :, :]], 'i_activations', summarize=20
    )

    # beta_v and beta_a: expand_dims to [N, OH, OW, 1, O, 1]
    beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
    beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]


    def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
      """The M-Step in EM Routing.

      :param rr: [N, OH, OW, KH x KW x I, O, 1],
        routing assignments from each input capsules (i) to each output capsules (o).
      :param votes: [N, OH, OW, KH x KW x I, O, PH x PW],
        input capsules poses x view transformation.
      :param i_activations: [N, OH, OW, KH x KW x I, 1, 1],
        input capsules activations, with dimensions expanded to match votes for broadcasting.
      :param beta_v: cost of describing capsules with one variance in each h-th compenents,
        should be learned discriminatively.
      :param beta_a: cost of describing capsules with one mean in across all h-th compenents,
        should be learned discriminatively.
      :param inverse_temperature: lambda, increase over steps with respect to a fixed schedule.

      :return: (o_mean, o_stdv, o_activation)
      """

      # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
      votes_shape = votes.get_shape().as_list()
      votes = tf.Print(
        votes, [votes.shape, votes[0, ..., :, 0, :]], 'mstep: votes', summarize=20
      )

      # rr_prime: [N, OH, OW, KH x KW x I, O, 1]
      rr_prime = rr * i_activations
      rr_prime = tf.Print(
        rr_prime, [rr_prime.shape, rr_prime[0, ..., :, 0, :]], 'mstep: rr_prime', summarize=20
      )

      # rr_prime_sum: sum acorss i, [N, OH, OW, 1, O, 1]
      rr_prime_sum = tf.reduce_sum(
        rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum'
      )
      rr_prime_sum = tf.Print(
        rr_prime_sum, [rr_prime_sum.shape, rr_prime_sum[0, ..., :, 0, :]], 'mstep: rr_prime_sum', summarize=20
      )

      # o_mean: [N, OH, OW, 1, O, PH x PW]
      o_mean = tf.reduce_sum(
        rr_prime * votes, axis=-3, keep_dims=True
      ) / rr_prime_sum
      o_mean = tf.Print(
        o_mean, [o_mean.shape, o_mean[0, ..., :, 0, :]], 'mstep: o_mean', summarize=20
      )

      # o_stdv: [N, OH, OW, 1, O, PH x PW]
      o_stdv = tf.sqrt(
        tf.reduce_sum(
          rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True
        ) / rr_prime_sum
      )
      o_stdv = tf.Print(
        o_stdv, [o_stdv.shape, o_stdv[0, ..., :, 0, :]], 'mstep: o_stdv', summarize=20
      )

      # o_cost: [N, OH, OW, 1, O, PH x PW]
      o_cost = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum
      o_cost = tf.Print(
        o_cost, [beta_v, o_cost.shape, o_cost[0, ..., :, 0, :]], 'mstep: beta_v, o_cost', summarize=20
      )

      # o_activation: [N, OH, OW, 1, O, 1]
      o_activations_cost = (beta_a - tf.reduce_sum(o_cost, axis=-1, keep_dims=True))
      # try to find a good inverse_temperature, for o_activation,
      o_activations_cost = tf.Print(
        o_activations_cost, [
          beta_a, inverse_temperature, o_activations_cost.shape, o_activations_cost[0, ..., :, 0, :]
        ], 'mstep: beta_a, inverse_temperature, o_activation_cost', summarize=20
      )
      o_activations = tf.sigmoid(
        inverse_temperature * o_activations_cost
      )
      o_activations = tf.Print(
        o_activations, [o_activations.shape, o_activations[0, ..., :, 0, :]], 'mstep: o_activation', summarize=20
      )

      return o_mean, o_stdv, o_activations

    def e_step(o_mean, o_stdv, o_activations, votes):
      """The E-Step in EM Routing.

      :param o_mean: [N, OH, OW, 1, O, PH x PW]
      :param o_stdv: [N, OH, OW, 1, O, PH x PW]
      :param o_activations: [N, OH, OW, 1, O, 1]
      :param votes: [N, OH, OW, KH x KW x I, O, PH x PW]

      :return: rr
      """

      # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
      votes_shape = votes.get_shape().as_list()
      votes = tf.Print(
        votes, [votes.shape, votes[0, ..., :, 0, :]], 'estep: votes', summarize=20
      )

      # o_p: [N, OH, OW, KH x KW x I, O, 1]
      # o_p is the probability density of the h-th component of the vote from i to c
      o_p_unit0 = - tf.reduce_sum(
        tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True
      )
      o_p_unit0 = tf.Print(
        o_p_unit0, [o_p_unit0.shape, o_p_unit0[0, ..., :, 0, :]], 'estep: o_p_unit0', summarize=20
      )
      # o_p_unit1 = - tf.log(
      #   0.50 * votes_shape[-1] * tf.log(2 * pi) + epsilon
      # )
      # o_p_unit1 = tf.Print(
      #   o_p_unit1, [o_p_unit1.shape, o_p_unit1[0, ..., :, 0, :]], 'estep: o_p_unit1', summarize=20
      # )
      o_p_unit2 = - tf.reduce_sum(
        tf.log(o_stdv + epsilon), axis=-1, keep_dims=True
      )
      o_p_unit2 = tf.Print(
        o_p_unit2, [o_p_unit2.shape, o_p_unit2[0, ..., :, 0, :]], 'estep: o_p_unit2', summarize=20
      )
      # o_p
      o_p = o_p_unit0 + o_p_unit2
      o_p = tf.Print(
        o_p, [o_p.shape, o_p[0, ..., :, 0, :]], 'estep: o_p', summarize=20
      )
      # rr: [N, OH, OW, KH x KW x I, O, 1]
      rr = tf.nn.softmax(
        tf.log(o_activations + epsilon) + o_p, dim=-2
      )
      rr = tf.Print(
        rr, [rr.shape, rr[0, ..., :, 0, :]], 'estep: rr', summarize=20
      )

      return rr

    for t in xrange(iterations):
      o_mean, o_stdv, o_activations = m_step(
        rr, votes, i_activations, beta_v, beta_a, inverse_temperature
      )
      if t < iterations - 1:
        rr = e_step(
          o_mean, o_stdv, o_activations, votes
        )

    # pose: [N, OH, OW, O, PH x PW] via squeeze o_mean [N, OH, OW, 1, O, PH x PW]
    poses = tf.squeeze(o_mean, axis=-3)

    # activation: [N, OH, OW, O] via squeeze o_activationis [N, OH, OW, 1, O, 1]
    activations = tf.squeeze(o_activations, axis=[-3, -1])

  return poses, activations

# ------------------------------------------------------------------------------#
# -------------------------------- capsules net --------------------------------#
# ------------------------------------------------------------------------------#

def capsule_net(inputs, num_classes, inverse_temperature, iterations, name='CapsuleEM-V0'):
  """"""
  with tf.variable_scope(name) as scope:

    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 1, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    nets = capsule_init(
      nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2 -> (poses, activations)
    nets = capsule_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_conv1'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    nets = capsule_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_conv2'
    )
    # inputs: (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view transform matrix within each channel -> (poses, activations)
    nets = capsule_fc(
      nets, num_classes, inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_fc'
    )

    poses, activations = nets

  return poses, activations

# ------------------------------------------------------------------------------#
# ------------------------------------ loss ------------------------------------#
# ------------------------------------------------------------------------------#

def spread_loss(labels, activations, margin, name):
  """This adds spread loss to total loss.

  :param labels: [N, O], where O is number of output classes, one hot vector, tf.uint8.
  :param activations: [N, O], activations.
  :param margin: margin 0.2 - 0.9 fixed schedule during training.

  :return: spread loss
  """

  activations_shape = activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)

    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )

    gap_mit = tf.reduce_sum(
      tf.square(
        tf.nn.relu(
          margin - (activations_t - activations_i)
        )
      )
    )

    # tf.add_to_collection(
    #   tf.GraphKeys.LOSSES, gap_mit
    # )
    #
    # total_loss = tf.add_n(
    #   tf.get_collection(
    #     tf.GraphKeys.LOSSES
    #   ), name='total_loss'
    # )

    tf.losses.add_loss(gap_mit)

    return gap_mit

# ------------------------------------------------------------------------------#

