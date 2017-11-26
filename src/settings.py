"""A script for matrix capsule with EM routing, settings for train/tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import mnist
from capsule import capsule


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'data_dir', 'data/mnist', 'Data Directory'
)
tf.app.flags.DEFINE_string(
  'train_dir', 'log/train', 'Train Directory.'
)
tf.app.flags.DEFINE_string(
  'tests_dir', 'log/tests', 'Tests Directory.'
)
tf.app.flags.DEFINE_string(
  'checkpoint_path', FLAGS.train_dir,
  'The directory where the model was written to or an absolute path to a checkpoint file.'
)

tf.app.flags.DEFINE_integer(
  'batch_size', 128, 'Train/Tests Batch Size.'
)

NUM_STEPS_PER_EPOCH = int(
  mnist.NUM_TRAIN_EXAMPLES / FLAGS.batch_size
)

slim = tf.contrib.slim
