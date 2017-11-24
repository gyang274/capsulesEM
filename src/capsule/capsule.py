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

  # TODO: vectorize the operation, or parallel this double for loop (the core covolution between capsule layers)

  # figure out the number of scan
  with tf.variable_scope(name) as scope:
    # init beta_v and beta_a
    beta_v = _get_weights_wrapper(name='beta_v', shape=[])
    beta_a = _get_weights_wrapper(name='beta_a', shape=[])
    # kernel: [1, KH, KW, I, O, PH, PW]
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights', shape=[1] + shape + inputs_pose_shape[-2:]
    )
    # note: tf.matmul doesn't support for broadcasting at this moment, manual tile to match with inputs_pose_slice_expansion on each dimension.
    # https://github.com/tensorflow/tensorflow/issues/216
    kernel = tf.tile(
      kernel, [inputs_pose_shape[0], 1, 1, 1, 1, 1, 1], name='pose_view_transform_weights_batch'
    )
    # apply same kernel to each (h_offset, w_offset) - weights share in convolution.
    pose = []; activation = []
    for h_offset in xrange(0, inputs_pose_shape[1] + 1 - shape[0], strides[1]):
      pose_wire = []; activation_wire = []
      for w_offset in xrange(0, inputs_pose_shape[2] + 1 - shape[1], strides[2]):
        vote_unit = []
        # at each offset K x K x I x PH x PW
        # inputs_pose_slice: [N, KH, KW, I, PH, PW]
        # slice inputs_pose [N, H, W, C, PH, PW] at (h_offset, w_offset) with KH x KW and with C == I
        inputs_pose_slice = inputs_pose[:, h_offset:(h_offset+shape[0]), w_offset:(w_offset+shape[1]), :, :, :]
        # inputs_pose_slice: [N, KH, KW, I, 1, PH, PW]
        # inputs_pose_slice_expansion: expand inputs_pose_slice dimension to match with kernel for broadcasting,
        # so that we can create output capsules at one offset (h_offset, w_offset) across all channels O at once.
        inputs_pose_slice_expansion = tf.expand_dims(
          inputs_pose_slice, axis=4, name='inputs_pose_slice_expansion'
        )
        # vote_unit: [N, KH, KW, I, O, PH, PW]
        # inputs_pose_slice_expansion matmul with kernel (view transform) becomes vote_unit
        # note: tf.matmul doesn't support for broadcasting at this moment, manual tile to match on each dimension.
        # https://github.com/tensorflow/tensorflow/issues/216
        vote_unit = tf.matmul(
          tf.tile(inputs_pose_slice_expansion, [1, 1, 1, 1, shape[-1], 1, 1]), kernel, name='pose_view_transform_vote'
        )
        vote_unit_shape = vote_unit.get_shape().as_list()
        # vote_unit: reshape into [N, KH, KW, I, O, PH x PW]
        vote_unit = tf.reshape(
          vote_unit, vote_unit_shape[:-2] + [vote_unit_shape[-2] * vote_unit_shape[-1]]
        )
        # vote_unit: reshape into [N, KH x KW x I, O, PH x PW]
        vote_unit = tf.reshape(
          vote_unit, [vote_unit_shape[0], vote_unit_shape[1] * vote_unit_shape[2] * vote_unit_shape[3], vote_unit_shape[4], vote_unit_shape[5] * vote_unit_shape[6]]
        )
        # inputs_activation_unit: [N, KH, KW, I]
        inputs_activation_unit = inputs_activation[:, h_offset:(h_offset+shape[0]), w_offset:(w_offset+shape[1]), :]
        # inputs_activation_unit: reshape into [N, KH x KW x I]
        inputs_activation_unit = tf.reshape(
          inputs_activation_unit, [inputs_activation_shape[0], shape[0] * shape[1] * inputs_activation_shape[-1]]
        )
        # vote_unit, inputs_activaiton_unit -> EM -> output_pose_unit, output_activation_unit
        # this operation involves inputs and output capsules at one (h_offset, w_offset) across all channels
        # pose_unit: [N, O, PH x PW]
        # activation_unit: [N, O]
        pose_unit, activation_unit = matrix_capsule_em_routing(
          vote_unit, inputs_activation_unit, beta_v, beta_a, inverse_temperature, iterations, name=name+'_em_'+str(h_offset)+'_'+str(w_offset)
        )
        # pose_unit: reshape to [N, O, PH, PW]
        pose_unit = tf.reshape(
          pose_unit, [vote_unit_shape[0]] + vote_unit_shape[4:]
        )
        pose_wire.append(pose_unit)
        activation_wire.append(activation_unit)
      # pose_wire: list with OW of [N, O, PH, PW] tensors, where OW is determined by W and strides
      pose.append(tf.stack(pose_wire, axis=1))
      # activation_wire: list with OW of [N, O] tensors, where OW is determined by W and strides
      activation.append(tf.stack(activation_wire, axis=1))
    # pose: list with OH of [N, OW, O, PH, PW] tensors, where OH is determinded by H and strides
    # pose: stack into [N, OH, OW, O, PH, PW]
    pose = tf.stack(pose, axis=1, name='pose')
    # activation: list with OH of [N, OW, O] tensors, where OH is determinded by H and strides
    # activation: stack into [N, OH, OW, O]
    activation = tf.stack(activation, axis=1, name='activation')

  return pose, activation


def matrix_capsule_em_routing(vote, i_activation, beta_v, beta_a, inverse_temperature, iterations, name):
  """The EM routing between input capsules (i) and output capsules (o).

  :param vote:
  :param i_activation:
  :param beta_v:
  :param beta_a:
  :param inverse_temperature:
  :param iterations:

  :return: (pose, activation) of output capsules.
  """

  # vote: [N, KH x KW x I, O, PH x PW]
  vote_shape = vote.get_shape().as_list()
  # i_activation: [N, KH x KW x I]
  i_activation_shape = i_activation.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # rr: [N, KH x KW x I, O, 1],
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(
      1.0/vote_shape[2], shape=vote_shape[:-1] + [1], dtype=tf.float32
    )
    # rr = tf.Print(
    #   rr, [rr.shape, rr[0, :, :, :]], 'rr', summarize=20
    # )

    # i_activation: expand to [N, KH x KW x I, 1, 1]
    i_activation = tf.expand_dims(
      tf.expand_dims(
        i_activation, axis=-1, name='i_activation_expansion_0'
      ), axis=-1, name='i_activation_expansion_1'
    )
    # i_activation = tf.Print(
    #   i_activation, [i_activation.shape, i_activation[0, :, :, :]], 'i_activation', summarize=20
    # )

    # note: match rr shape, i_activation shape with shape vote for broadcasting in EM

    def m_step(rr, vote, i_activation, beta_v, beta_a, inverse_temperature):
      """The M-Step in EM Routing.

      :param rr: [N, KH x KW x I, O, 1] routing from each input capsules (i) to each output capsules (o).
      :param vote: [N, KH x KW x I, O, PH x PW] input capsules pose x view transformation.
      :param i_activation: [N, KH x KW x I, 1, 1] input capsules activations, with dimensions expanded to match vote for broadcasting.
      :param beta_v: constant, cost of describing capsules with one variance in each h-th compenents, should be learned discriminatively.
      :param beta_a: constant, cost of describing capsules with one mean in across all h-th compenents, should be learned discriminatively.
      :param inverse_temperature: lambda, increase at each iteration with a fixed schedule.

      :return: (o_mean, o_stdv, o_activation)
      """

      # rr_prime: [N, KH x KW x I, O, 1]
      rr_prime = rr * i_activation
      # rr_prime = tf.Print(
      #   rr_prime, [rr_prime.shape, rr_prime[0, :, 0, :]], 'mstep: rr_prime', summarize=20
      # )

      # rr_prime_sum: sum acorss i, [N, 1, O, 1]
      rr_prime_sum = tf.reduce_sum(
        rr_prime, axis=1, keep_dims=True, name='rr_prime_sum'
      )
      # rr_prime_sum = tf.Print(
      #   rr_prime_sum, [rr_prime_sum.shape, rr_prime_sum[0, :, :, :]], 'mstep: rr_prime_sum', summarize=20
      # )

      # vote: [N, KH x KW x I, O, PH x PW]
      # rr_prime: [N, KH x KW x I, O, 1]
      # rr_prime_sum: [N, 1, O, 1]
      # o_mean: [N, 1, O, PH x PW]
      o_mean = tf.reduce_sum(
        rr_prime * vote, axis=1, keep_dims=True
      ) / rr_prime_sum
      # o_mean = tf.Print(o_mean, [o_mean.shape, o_mean[0, :, :, :]], 'mstep: o_mean', summarize=20)
      # o_stdv: [N, 1, O, PH x PW]
      o_stdv = tf.sqrt(
        tf.reduce_sum(
          rr_prime * tf.square(vote - o_mean), axis=1, keep_dims=True
        ) / rr_prime_sum
      )
      # o_stdv = tf.Print(o_stdv, [o_stdv.shape, o_stdv[0, :, :, :]], 'mstep: o_stdv', summarize=20)
      # o_cost: [N, 1, O, PH x PW]
      o_cost = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum
      # o_cost = tf.Print(o_cost, [beta_v, o_cost.shape, o_cost[0, :, :, :]], 'mstep: beta_v, o_cost', summarize=20)
      # o_activation: [N, 1, O, 1]
      o_activation_cost = (beta_a - tf.reduce_sum(o_cost, axis=-1, keep_dims=True))
      # try to find a good inverse_temperature, for o_activation,
      # o_activation_cost = tf.Print(
      #   o_activation_cost, [inverse_temperature, beta_a, o_activation_cost.shape, o_activation_cost[0, :, :, :]], 'mstep: inverse_temperature, beta_a, o_activation_cost', summarize=20
      # )
      o_activation = tf.sigmoid(
        inverse_temperature * o_activation_cost
      )
      # o_activation = tf.Print(o_activation, [o_activation.shape, o_activation[0, :, :, :]], 'mstep: o_activation', summarize=20)

      return o_mean, o_stdv, o_activation

    def e_step(o_mean, o_stdv, o_activation, vote):
      """The E-Step in EM Routing.

      :param o_mean: [N, 1, O, PH x PW]
      :param o_stdv: [N, 1, O, PH x PW]
      :param o_activation: [N, 1, O, 1]
      :param vote: [N, KH x KW x I, O, PH x PW]

      :return: rr
      """

      # vote: [N, KH x KW x I, O, PH x PW]
      vote_shape = vote.get_shape().as_list()
      # vote = tf.Print(vote, [vote.shape, vote[0, :, :, :]], 'estep: vote', summarize=20)
      # o_p: [N, KH x KW x I, O, 1]
      # o_p is the probability density of the h-th component of the vote from i to c

      # o_p: version 0
      o_p_unit0 = - tf.reduce_sum(
        tf.square(vote - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True
      )
      # o_p_unit0 = tf.Print(o_p_unit0, [o_p_unit0.shape, o_p_unit0[0, :, :, :]], 'estep: o_p_unit0', summarize=20)
      # o_p_unit1 = - tf.log(
      #   0.50 * vote_shape[-1] * tf.log(2 * pi) + epsilon
      # )
      # o_p_unit1 = tf.Print(o_p_unit1, [o_p_unit1.shape, o_p_unit1], 'estep: o_p_unit1', summarize=20)
      o_p_unit2 = - tf.reduce_sum(
        tf.log(o_stdv + epsilon), axis=-1, keep_dims=True
      )
      # o_p_unit2 = tf.Print(o_p_unit2, [o_p_unit2.shape, o_p_unit2[0, :, :, :]], 'estep: o_p_unit2', summarize=20)
      # o_p = tf.exp(
      #   o_p_unit0 + o_p_unit1 + o_p_unit2
      # )
      # o_p = tf.Print(o_p, [o_p.shape, o_p[0, :, :, :]], 'estep: o_p', summarize=20)
      # # rr: [N, KH x KW x I, O, 1]
      # rr = o_activation * o_p
      # rr = tf.Print(rr, [rr.shape, rr[0, :, :, :]], 'estep: rr before division', summarize=20)
      # rr = rr / tf.reduce_sum(rr, axis=2, keep_dims=True)
      # rr = tf.Print(rr, [rr.shape, rr[0, :, :, :]], 'estep: rr after division', summarize=20)

      # o_p: version 1: numerical stable
      o_p = o_p_unit0 + o_p_unit2
      # o_p = tf.Print(o_p, [o_p.shape, o_p[0, :, :, :]], 'estep: o_p', summarize=20)
      rr = tf.log(o_activation + epsilon) + o_p
      # rr = tf.Print(rr, [rr.shape, rr[0, :, :, :]], 'estep: rr before softmax', summarize=20)
      rr = tf.nn.softmax(rr, dim=2)
      # rr = tf.Print(rr, [rr.shape, rr[0, :, :, :]], 'estep: rr after softmax', summarize=20)

      return rr

    for t in xrange(iterations):
      o_mean, o_stdv, o_activation = m_step(
        rr, vote, i_activation, beta_v, beta_a, inverse_temperature
      )
      if t < iterations - 1:
        rr = e_step(
          o_mean, o_stdv, o_activation, vote
        )

    # pose: [N, O, PH x PW]
    pose = tf.squeeze(o_mean, axis=1)

    # activation: [N, O]
    activation = tf.squeeze(o_activation, axis=[1, -1])

  return pose, activation


def capsule_fc(inputs, num_classes, inverse_temperature, iterations, name):
  """This constructs a fully-connected layer from convolution capsule layer to output capsule layer.

  :param inputs: a primary or convolution capsule layer with pose and activation,
    pose shape [N, H, W, C, PH, PW], activation shape [N, H, W, C]
  :param num_classes: number of classes.

  :return: (pose, activation)

  note:
    This is the D -> E in paper.
    This step includes two major sub-steps:
      1. Apply a shared view transform weight matrix PH x PW (4 x 4) to the same input channel.
        (This is why in D it has 1 x 1, and why the weight are D x E x 4 x 4)
      2. Re-struct the inputs vote from [N, H, W, I, PH, PW] into [N, H x W x I, PH x PW],
        add scaled coordinate on first two elements, EM routing an output [N, O, PH x PW],
        and reshape output [N, O, PH, PW].
    The difference between fully-connected layer and convolution layer, is that:
      1. The corresponding KH, KW in this fully-connected here is actually the whole H, W, not 1, 1.
      2. The view transformation matrix is shared within KH, KW (i.e., H, W) in this fully-connected layer,
        whereas in the convolution capsule layer, the view transformation can be different for each capsule
        in the KH, KW, but shared across (KH, KW, I, O, PH, PW).
  """

  inputs_pose, inputs_activation = inputs

  inputs_pose_shape = inputs_pose.get_shape().as_list()

  inputs_activation_shape = inputs_activation.get_shape().as_list()

  with tf.variable_scope(name) as scope:
    # init beta_v and beta_a
    beta_v = _get_weights_wrapper(name='beta_v', shape=[])
    beta_a = _get_weights_wrapper(name='beta_a', shape=[])
    # note: in paper:
    # share the transformation matrices between different positions of the same capsule type,
    # and add the scaled coordinate (row, column) of the center of the receptive field of each capsule to the first two elements of its vote
    # kernel: [1, 1, 1, I, O, PH, PW]
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights',
      shape=[1, 1, 1, inputs_pose_shape[3], num_classes] + inputs_pose_shape[-2:],
      weights_decay_factor=0.0
    )
    kernel = tf.tile(
      kernel, inputs_pose_shape[0:3]+ [1, 1, 1, 1], name='pose_view_transform_weights_batch'
    )
    # inputs_pose_expansion: [N, H, W, I, 1, PH, PW]
    # inputs_pose_expansion: expand inputs_pose dimension to match with kernel for broadcasting,
    # share the transformation matrices as kernel (1, 1) broadcasting to inputs pose expansion (H, W)
    inputs_pose_expansion = tf.expand_dims(
      inputs_pose, axis=4, name='inputs_pose_expansion'
    )
    # vote: [N, H, W, I, O, PH, PW]
    vote = tf.matmul(
      tf.tile(inputs_pose_expansion, [1, 1, 1, 1, num_classes, 1, 1]), kernel, name='pose_view_transform_vote'
    )
    vote_shape = vote.get_shape().as_list()
    # vote: reshape into [N, H, W, I, O, PH x PW]
    vote = tf.reshape(
      vote, vote_shape[:-2] + [vote_shape[-2] * vote_shape[-1]]
    )

    # add scaled coordinate
    H = inputs_pose_shape[1]
    W = inputs_pose_shape[2]

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

    vote = vote + coordinate_offset_h + coordinate_offset_w

    # vote: reshape into [N, H x W x I, O, PH x PW]
    vote = tf.reshape(
      vote, [vote_shape[0], vote_shape[1] * vote_shape[2] * vote_shape[3], vote_shape[4], vote_shape[5] * vote_shape[6]]
    )

    # inputs_activation: [N, H, W, I]
    # inputs_activation: reshape into [N, KH x KW x I]
    inputs_activation = tf.reshape(
      inputs_activation, [inputs_activation_shape[0], inputs_activation_shape[1] * inputs_activation_shape[2] * inputs_activation_shape[3]]
    )
    # vote, inputs_activaiton_unit -> EM -> output_pose_unit, output_activation_unit
    # pose: [N, O, PH x PW]
    # activation: [N, O]
    pose, activation = matrix_capsule_em_routing(
      vote, inputs_activation, beta_v, beta_a, inverse_temperature, iterations, name=name + '_em_'
    )
    # pose: reshape to [N, O, PH, PW]
    pose = tf.reshape(
      pose, [vote_shape[0]] + vote_shape[4:]
    )

  return pose, activation

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
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32 x (4x4+1) -> pose, activation
    nets = capsule_init(
      nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
    )
    # inputs: (pose, activation) -> capsule-conv 3x3x32x32x4x4 -> (pose, activation)
    nets = capsule_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_conv1'
    )
    # inputs: (pose, activation) -> capsule-conv 3x3x32x32x4x4 -> (pose, activation)
    nets = capsule_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_conv2'
    )
    # inputs: (pose, activation) -> capsule-fc HxW share weight between 1x1 -> (pose, activation)
    nets = capsule_fc(
      nets, num_classes, inverse_temperature=inverse_temperature, iterations=iterations, name='capsule_fc'
    )

    pose, activation = nets

  return pose, activation

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

