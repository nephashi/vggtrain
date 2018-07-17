import tensorflow as tf
import model
import vgg_input
import time
from datetime import datetime
import constant

LOG_FREQUENCY = constant.LOG_FREQUENCY
BATCH_SIZE = constant.BATCH_SIZE
TRAIN_DIR = constant.TRAIN_DIR
MAX_STEP = constant.MAX_STEP

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.device('/cpu:0'):
            images, labels = vgg_input.input('E:\\train_200.tfrecords')
        logits = model.network(images)
        loss = model.loss(logits, labels)
        tf.summary.scalar('loss', loss)
        train_op = tf.train.RMSPropOptimizer(1e-4).minimize(loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('model_summary')

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % LOG_FREQUENCY == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
                    sec_per_batch = float(duration / LOG_FREQUENCY)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

        # step = 0
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     while True:
        #         _, loss = sess.run([train_op, loss])
        #         if (step != 0 and step % 10 == 0):
        #             print("step " + repr(step) + ", loss " + repr(loss))
        #             summary = sess.run(merged)
        #             train_writer.add_summary(summary, step)
        #         step += 1

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=TRAIN_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=MAX_STEP),
                        tf.train.NanTensorHook(loss),
                        _LoggerHook()],
                config=tf.ConfigProto(
                        log_device_placement=False)
                # save_summaries_steps=None,
                # save_summaries_secs=None
                ) as mon_sess:

            step = 0

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                # step += 1
                # if(step % 100 == 0):
                #     print("writing summary")
                #     summary = mon_sess.run(merged)
                #     train_writer.add_summary(summary, step)

train()