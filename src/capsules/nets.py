"""An implementation of matrix capsules with EM routing.
"""

import tensorflow as tf

from core import _conv2d_wrapper, capsules_init, capsules_conv, capsules_fc

slim = tf.contrib.slim

# ------------------------------------------------------------------------------#
# -------------------------------- capsules net --------------------------------#
# ------------------------------------------------------------------------------#

def capsules_v0(inputs, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 1, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    nets = capsules_init(
      nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2 -> (poses, activations)
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv2'
    )
    # inputs: (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view transform matrix within each channel -> (poses, activations)
    nets = capsules_fc(
      nets, num_classes, iterations=iterations, name='capsule_fc'
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

    # margin = tf.Print(
    #   margin, [margin], 'margin', summarize=20
    # )

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

