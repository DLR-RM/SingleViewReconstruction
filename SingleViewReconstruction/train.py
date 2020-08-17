import tensorflow as tf
from src.TreeModel import TreeModel
from src.DataSetLoader import DataSetLoader
from src.LossManager import LossManager
from src.SettingsReader import SettingsReader
import os
from src.utils import StopWatch
from src.utils import StopWatchContext
from src.utils import TimeFrame
from src.utils import AverageNumber


class ReconstructioNetworkTrainer(object):

    def __init__(self, settings_path):
        print("Settings Path: {}".format(settings_path))
        self._settings_file_path = settings_path
        # read all settings used to construct the model and the data loader
        self._settings = SettingsReader(settings_path)

        # configure the data set loader
        self._loader = DataSetLoader(self._settings)
        input_to_network = self._loader.get_input()
        self._train_init_op = self._loader.train_init_op
        self._test_init_op = self._loader.test_init_op
        self._validation_init_op = self._loader.validation_init_op

        # create the full model
        with StopWatchContext("Needed for creation of tree model:") as swc:
            self._model = TreeModel(input=input_to_network, settings=self._settings)
        self._rescaled_layer_before_3d = self._model.rescaled_layer_before_3D
        self._last_layer = self._model.last_layer

        # create the loss manager
        self._loss_manager = LossManager(self._model, self._settings, self._loader)
        self._loss_manager.generate_loss()
        self._train_loss = self._loss_manager.trainings_loss

        self._create_optimizer()

        # variables used for training
        self._saver = tf.train.Saver()
        self._train_dict, self._test_dict = {}, {}
        self._use_test_set = True
        self._small_batch_counter = 0
        self._test_batch_counter = 0

    def _create_optimizer(self):
        with tf.name_scope('training_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            learning_rate = self._settings.learning_rate
            self._optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4)
            self._train_op = tf.group([self._optimizer.minimize(self._train_loss), update_ops])

    def start_training(self):
        config = tf.ConfigProto()
        # if settings.is_home_desktop:
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as self._sess:
            with StopWatchContext("Creating the writer in: ") as sw:
                self._train_writer = tf.summary.FileWriter(os.path.join(self._settings.LOG_DIR, 'train'),
                                                           self._sess.graph, flush_secs=1)
                self._test_writer = tf.summary.FileWriter(os.path.join(self._settings.LOG_DIR, 'test'), flush_secs=1)

            with StopWatchContext("Done vars in: ", "Init vars") as sw:
                self._sess.run(tf.global_variables_initializer())
            self._sess.graph.finalize()

            if self._use_test_set:
                self._sess.run(self._test_init_op)

                self._loss_manager.run_initial(self._sess, self._train_writer, self._small_batch_counter,
                                               self._train_dict)

            self._test_timer = StopWatch()
            self._save_timer = StopWatch()
            self._train_sum_timer = StopWatch()
            self._model_timer = StopWatch()

            self._sess.run(self._train_init_op)
            self._old_loss_val = 100
            self._training_ops = [self._train_op, self._train_loss]
            self._training_ops_plus_summary = [self._train_op, self._train_loss]
            self._training_ops_plus_summary.extend(self._loss_manager.get_summaries())
            self._avg_trainings_time = AverageNumber()
            self._cycle_avg_time = AverageNumber()

            # iterating over the dataset
            for i in range(1000000):
                self._train_fct(i)

            self._train_writer.close()
            self._test_writer.close()

    def _perform_test_writer_update(self):
        if self._test_timer.elapsed_time_val > 360.0 and self._use_test_set and self._test_batch_counter > 1000:
            self._test_batch_counter = 0
            with StopWatchContext("Needed for summaries:") as swc:
                with StopWatchContext("Changed to test data in:") as swc2:
                    self._sess.run(self._test_init_op)
                current_test_loss = self._loss_manager.run_summary(self._sess, self._test_writer,
                                                                   self._small_batch_counter, self._test_dict)
                with open(os.path.join(self._settings.LOG_DIR, 'loss_file.txt'), 'w') as file:
                    file.write(str(self._old_loss_val) + "\n" + str(current_test_loss))
            with StopWatchContext("Changed to train data in:") as swc:
                self._sess.run(self._train_init_op)
            self._test_timer.reset()
            return True
        return False

    def _train_fct(self, i):
        cycle_sw = StopWatch()
        change_data_set = self._perform_test_writer_update()
        self._test_batch_counter += 1
        if self._model_timer.elapsed_time_val > 10 * 60.:
            with StopWatchContext("Saving took:") as sw:
                self._saver.save(self._sess, os.path.join(self._settings.LOG_DIR, 'model.ckpt'))
            self._model_timer.reset()
            change_data_set = True

        if self._train_sum_timer.elapsed_time_val < 120.:
            used_trainings_ops = self._training_ops
        else:
            used_trainings_ops = self._training_ops_plus_summary
            self._train_sum_timer.reset()
        sw_training = StopWatch()
        try:
            trainings_output_res = self._sess.run(used_trainings_ops, self._train_dict)
        except tf.errors.ResourceExhaustedError:
            print("Batch size to big: {}".format(self._settings.batch_size))
            exit(1)
        needed_for_training = sw_training.elapsed_time_val
        loss_res_value = trainings_output_res[1]
        if i != 0 and not change_data_set:
            self._avg_trainings_time.add_new(needed_for_training)
        loss_string = "{:.9f}".format(loss_res_value)
        print("Time training: {}, loss: {}, at: \t{}, avg time: {}, cycle: {}".format(TimeFrame(needed_for_training),
                                                                                      loss_string, i, TimeFrame(
                self._avg_trainings_time.mean), TimeFrame(self._cycle_avg_time.mean)))
        special_loss = self._old_loss_val * 300.0
        if loss_res_value < special_loss:
            for sum in trainings_output_res[2:]:
                self._train_writer.add_summary(sum, global_step=self._small_batch_counter)
        elif self._small_batch_counter > 10 and loss_res_value > special_loss:
            print("Special gigant loss: {}".format(loss_res_value))
            self._train_writer.close()
            self._test_writer.close()
            exit(0)

        self._old_loss_val = loss_res_value
        if self._small_batch_counter > 1:
            self._cycle_avg_time.add_new(cycle_sw.elapsed_time_val)
        self._small_batch_counter += 1


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    settings_file_path = os.path.abspath('settings_file.yml')
    reconstruction_network_trainer = ReconstructioNetworkTrainer(settings_file_path)

    reconstruction_network_trainer.start_training()
