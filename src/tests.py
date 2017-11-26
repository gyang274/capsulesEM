"""A tests script for matrix capsule with EM routing."""

from settings import *


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default(), tf.device('/cpu:0'):

    global_step = slim.get_or_create_global_step()
    global_step = tf.Print(
      global_step, [global_step], 'global_step'
    )

    images, labels = mnist.inputs(
      data_directory=FLAGS.data_dir,
      is_training=False,
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

    # margin = tf.train.piecewise_constant(
    #   tf.cast(global_step, dtype=tf.int32),
    #   boundaries=[
    #     int(NUM_STEPS_PER_EPOCH * 50 * x / 7) for x in xrange(1, 8)
    #   ],
    #   values=[
    #     x / 10.0 for x in range(2, 10)
    #   ]
    # )

    poses, activations = capsule.capsule_net(
      images,
      num_classes=10,
      inverse_temperature=inverse_temperature,
      iterations=3,
      name='CapsuleEM-V0'
    )

    variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(activations, 1)

    labels = tf.argmax(labels, 1)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
      {
        'Accuracy': slim.metrics.streaming_accuracy(
          predictions, labels
        )
      }
    )

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    # yg: use num_epochs = 1 in data.py: load_batch: slim.dataset_data_provider.DatasetDataProvider
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    # yg: move to evaluation_loop from evaluate_once
    # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    # else:
    #   checkpoint_path = FLAGS.checkpoint_path
    #
    # tf.logging.info('Evaluating %s' % checkpoint_path)
    #
    # slim.evaluation.evaluate_once(
    #     master=FLAGS.master,
    #     checkpoint_path=checkpoint_path,
    #     # logdir=FLAGS.eval_dir,
    #     logdir=FLAGS.tests_dir,
    #     num_evals=num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     variables_to_restore=variables_to_restore,
    #     # yg: add session_config to limit gpu usage and allow growth
    #     session_config=tf.ConfigProto(
    #       # device_count = {
    #       #   'GPU': 0
    #       # },
    #       gpu_options={
    #         'allow_growth': 1,
    #         # 'per_process_gpu_memory_fraction': 0.01
    #       },
    #       allow_soft_placement=True,
    #       log_device_placement=False
    #     )
    # )
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      raise ValueError('You must supply the checkpoint path as a directory contains training checkpoints.')

    slim.evaluation.evaluation_loop(
      master='',
      checkpoint_dir=FLAGS.checkpoint_path,
      # logdir=FLAGS.eval_dir,
      logdir=FLAGS.tests_dir,
      # num_evals=num_batches, # use num_epochs in data.py
      eval_op=list(names_to_updates.values()),
      variables_to_restore=variables_to_restore,
      # move to evaluation_loop from evaluate_once
      eval_interval_secs=10,
      max_number_of_evaluations=None,
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

if __name__ == '__main__':
  tf.app.run()
