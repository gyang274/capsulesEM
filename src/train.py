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
      data_directory=FLAGS.data_dir, is_training=True, batch_size=FLAGS.batch_size
    )

    poses, activations = capsules.nets.capsules_v0(
      images, num_classes=10, iterations=1, name='capsulesEM-V0'
    )
    # activations = tf.Print(
    #   activations, [activations.shape, activations[0, ...]], 'activations', summarize=20
    # )

    # inverse_temperature = tf.train.piecewise_constant(
    #   tf.cast(global_step, dtype=tf.int32),
    #   boundaries=[
    #     int(NUM_STEPS_PER_EPOCH * 10),
    #     int(NUM_STEPS_PER_EPOCH * 20),
    #     int(NUM_STEPS_PER_EPOCH * 30),
    #     int(NUM_STEPS_PER_EPOCH * 50),
    #   ],
    #   values=[0.001, 0.001, 0.002, 0.002, 0.005]
    # )

    # margin schedule
    # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
    margin_schedule_epoch_achieve_max = 10.0
    margin = tf.train.piecewise_constant(
      tf.cast(global_step, dtype=tf.int32),
      boundaries=[
        int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in xrange(1, 8)
      ],
      values=[
        x / 10.0 for x in range(2, 10)
      ]
    )

    # loss = tf.reduce_sum(
    #   tf.nn.softmax_cross_entropy_with_logits(
    #     labels=labels, logits=activations, name='cross_entropy_loss'
    #   )
    # )
    loss = capsules.nets.spread_loss(
      labels, activations, margin=margin, name='spread_loss'
    )

    # tf.summary.scalar(
    #   'losses/cross_entropy_loss', loss
    # )
    tf.summary.scalar(
      'losses/spread_loss', loss
    )

    # TODO: set up a learning_rate decay
    optimizer = tf.train.AdamOptimizer(
      learning_rate=0.001
    )

    # grads_and_vars = optimizer.compute_gradients(loss)
    # for (grad, var) in grads_and_vars:
    #   grad = tf.Print(
    #     grad, [grad.shape, grad], str(grad.name), summarize=20
    #   )
    #   var = tf.Print(
    #     var, [var.shape, var], str(var.name), summarize=20
    #   )
    # train_op = optimizer.apply_gradients(grads_and_vars)

    train_tensor = slim.learning.create_train_op(
      loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
    )

    slim.learning.train(
      train_tensor,
      logdir=FLAGS.train_dir,
      log_every_n_steps=10,
      save_summaries_secs=60,
      saver=tf.train.Saver(max_to_keep=100),
      save_interval_secs=600,
      # yg: add session_config to limit gpu usage and allow growth
      session_config=tf.ConfigProto(
        # device_count = {
        #   'GPU': 0
        # },
        gpu_options={
          'allow_growth': 0,
          # 'per_process_gpu_memory_fraction': 0.01
          'visible_device_list': '0'
        },
        allow_soft_placement=True,
        log_device_placement=False
      )
    )
    # init = tf.global_variables_initializer()
    # sess = tf.Session(
    #   config=tf.ConfigProto(
    #     # device_count = {
    #     #   'GPU': 0
    #     # },
    #     gpu_options={
    #       'allow_growth': 0,
    #       # 'per_process_gpu_memory_fraction': 0.01
    #       'visible_device_list': '0'
    #     },
    #     allow_soft_placement=True,
    #     log_device_placement=False
    #   )
    # )
    # sess.run(init)
    #
    # tf.train.start_queue_runners(sess=sess)
    #
    # for step in range(1001):
    #   print('step: ', step)
    #   _, loss_value = sess.run([train_op, loss])
    #   print('step: ', step, 'loss: ', loss_value)
    #   if step % 10 == 0:
    #     print('gv')
    #     for gv in grads_and_vars:
    #       print(gv[1])
    #       print(str(sess.run(gv[1].name)))
    #       z = [0 for x in gv[0].shape]
    #       print(sess.run(gv[0][z]))


if __name__ == "__main__":
  tf.app.run()

