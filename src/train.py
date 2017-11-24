"""A train script for matrix capsule with EM routing."""

# TODO: re-write use generic sess/train, without slim.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import mnist
from capsule import capsule


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
  'batch_size', 128, 'Train Batch Size.'
)
tf.app.flags.DEFINE_string(
  'data_dir', 'data/mnist', 'Data Directory'
)
tf.app.flags.DEFINE_string(
  'train_dir', 'log/train', 'Train Directory.'
)

NUM_STEPS_PER_EPOCH = int(
  mnist.NUM_TRAIN_EXAMPLES / FLAGS.batch_size
)

slim = tf.contrib.slim

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images, labels = mnist.inputs(
        data_directory=FLAGS.data_dir,
        is_training=True,
        batch_size=FLAGS.batch_size
      )
      inverse_temperature = tf.train.piecewise_constant(
        tf.cast(global_step, dtype=tf.int32),
        boundaries=[
          int(NUM_STEPS_PER_EPOCH * 10),
          int(NUM_STEPS_PER_EPOCH * 20),
          int(NUM_STEPS_PER_EPOCH * 30),
          int(NUM_STEPS_PER_EPOCH * 50),
        ],
        values=[0.001, 0.002, 0.005, 0.010, 0.020]
      )
      margin = tf.train.piecewise_constant(
        tf.cast(global_step, dtype=tf.int32),
        boundaries=[
          int(NUM_STEPS_PER_EPOCH * 50 * x / 7) for x in xrange(1, 8)
        ],
        values=[
          x / 10.0 for x in range(2, 10)
        ]
      )
      poses, activations = capsule.capsule_net(
        images,
        num_classes=10,
        inverse_temperature=inverse_temperature,
        iterations=3,
        name='CapsuleEM-V0'
      )
      loss = capsule.spread_loss(
        labels, activations, margin=margin, name='spread_loss'
      )
      tf.summary.scalar(
        'losses/spread_loss', loss
      )
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

      train_tensor = slim.learning.create_train_op(
        loss, optimizer
      )

      slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        log_every_n_steps=10,
        save_summaries_secs=20,
        saver=tf.train.Saver(max_to_keep=100),
        save_interval_secs=20,
        # yg: add session_config to limit gpu usage and allow growth
        session_config=tf.ConfigProto(
          # device_count = {
          #   'GPU': 0
          # },
          gpu_options={
            'allow_growth': 1,
            # 'per_process_gpu_memory_fraction': 0.01
          },
          allow_soft_placement=True,
          log_device_placement=False
        )
      )

if __name__ == "__main__":
  tf.app.run()

