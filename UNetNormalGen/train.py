import os
import tensorflow as tf
import datetime

from source.loss_manager import LossManager
from source.data_loader import DataLoader
from source.settings_reader import SettingsReader
from source.model import Model

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
single_view_path = os.path.abspath(os.path.join(main_path, "SingleViewReconstruction"))
import sys
sys.path.append(main_path)
sys.path.append(single_view_path)

from src.utils import StopWatch


settings_file_path = os.path.join(os.path.dirname(__file__), "settings", "settings_file.yml")
settings = SettingsReader(settings_file_path)

data_loader = DataLoader(settings)

# Logging
time_str = str(datetime.datetime.now())
time_str = time_str.replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
log_dir = os.path.join("logs", time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

validation_size = int(settings.validation_ratio * settings.max_dataset_size)
validation_steps = int(validation_size // settings.batch_size)

train_steps = int((settings.max_dataset_size - validation_size) // settings.batch_size)

# Dataset iterators
trn_op, val_op = data_loader.load_default_iterator()
x_iter, y_iter = trn_op.get_next()
x_iter_val, y_iter_val = val_op.get_next()

val_bool = tf.placeholder(dtype=bool, shape=())
data = tf.cond(val_bool, lambda: x_iter, lambda: x_iter_val)
ground_truth = tf.cond(val_bool, lambda: y_iter, lambda: y_iter_val)
tf.summary.image('ground truth', (ground_truth + 1.) / 2. * 255.)
tf.summary.image('color', data * 255.)

# create the model
model = Model()
model_result = model.create(data)


# LossManager
last_layer, _, _, _ = model.get_results()
loss_manager = LossManager(ground_truth, last_layer)
loss = loss_manager.cosine_similarity()
op, cost = model.compile(settings.learning_rate, loss)

# Timers
model_timer = StopWatch()
train_sum_timer = StopWatch()
val_sum_timer = StopWatch()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run([trn_op.initializer, val_op.initializer])

    # Writers
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph, flush_secs=10)
    test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'), sess.graph, flush_secs=10)
    tf.io.write_graph(sess.graph, log_dir, 'graph.pbtxt')

    # operations
    merged = tf.summary.merge_all()
    training_ops = [cost, op]
    training_ops_plus_summary = [merged, cost, op]

    # Saver
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    min_for_test = 15
    global_step = 0
    try:    
        for epoch_i in range(9999999):
            for i in range(train_steps):
                print("Current step: {}".format(global_step))
                if train_sum_timer.elapsed_time_val > min_for_test * 60:
                    train_sum_timer.reset()
                    trainings_output_res = sess.run(training_ops_plus_summary, feed_dict={val_bool: True})
                    train_writer.add_summary(trainings_output_res[0], global_step)
                else:
                    trainings_output_res = sess.run(training_ops, feed_dict={val_bool: True})

                if val_sum_timer.elapsed_time_val > min_for_test * 60.:
                    val_sum_timer.reset()
                    summary, _ = sess.run([merged, cost], feed_dict={val_bool: False})
                    test_writer.add_summary(summary, global_step)

                if model_timer.elapsed_time_val > 3*60*60:
                    model_timer.reset()
                    saver.save(sess, os.path.join(log_dir, 'model.ckpt'))
                global_step += 1

    except tf.errors.ResourceExhaustedError:
        print("Batch size too big: " + str(settings.batch_size))
        exit(1)
