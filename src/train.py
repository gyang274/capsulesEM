"""A train script for matrix capsule with EM routing."""

from settings import *

def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  NUM_STEPS_PER_EPOCH = int(
    mnist.NUM_TRAIN_EXAMPLES / FLAGS.batch_size
  )

  with tf.Graph().as_default():

    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()

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
      values=[0.001, 0.001, 0.002, 0.002, 0.005]
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

    poses, activations = capsule.nets.capsule_net(
      images,
      num_classes=10,
      inverse_temperature=inverse_temperature,
      iterations=3,
      name='CapsuleEM-V0'
    )

    loss = capsule.nets.spread_loss(
      labels, activations, margin=margin, name='spread_loss'
    )

    tf.summary.scalar(
      'losses/spread_loss', loss
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    train_tensor = slim.learning.create_train_op(
      loss, optimizer, global_step=global_step
    )

    slim.learning.train(
      train_tensor,
      logdir=FLAGS.train_dir,
      log_every_n_steps=1,
      save_summaries_secs=240,
      saver=tf.train.Saver(max_to_keep=100),
      save_interval_secs=600,
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

