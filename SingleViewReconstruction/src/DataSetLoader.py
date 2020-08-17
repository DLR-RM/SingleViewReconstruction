import tensorflow as tf
import numpy as np
import os
import h5py
import cv2
import glob

class DataSetLoader(object):

    def __init__(self, settings):
        self.settings = settings
        self.amount_of_output_channels = self.settings.amount_of_output_channels
        autotune = tf.data.experimental.AUTOTUNE

        # for denormalizing, are channel last
        own_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        with h5py.File(os.path.join(own_data_folder, 'color_normal_mean.hdf5'), 'r') as data:
            self.mean_img = np.array(data["color"])
            self.normal_mean_img = np.array(data["normal"])

        settings.data_file_paths.sort()
        test_files = settings.data_file_paths[:2]
        print(test_files)
        settings.data_file_paths = settings.data_file_paths[2:]

        self.train_dataset = self._get_dataset_from_path(settings.data_file_paths)
        self.test_dataset = self._get_dataset_from_path(test_files, settings.test_shuffle_paths)

        validation_size = 1
        self.validation_dataset = self.test_dataset.take(validation_size)
        self.test_dataset = self.test_dataset.skip(validation_size)
        #self.validation_dataset = self.validation_dataset.cache()
        self.validation_dataset = self.validation_dataset.map(lambda color_img, normal_img, voxel, lossmap: (self.norm_color_img(color_img), normal_img, voxel, lossmap))
        self.validation_dataset = self.validation_dataset.batch(validation_size)
        self.validation_dataset = self.validation_dataset.repeat()

        if settings.test_dataset_size > 0:
            # saves the whole dataset in the memory, to increase the speed
            self.test_dataset = self.test_dataset.repeat()
            self.test_dataset = self.test_dataset.map(lambda color_img, normal_img, voxel, lossmap: (self.norm_color_img(color_img), normal_img, voxel, lossmap))
            self.test_dataset = self.test_dataset.batch(settings.test_batch_size)
        else:
            self.test_dataset = None

        shuffle_size = settings.shuffle_size
        max_train_dataset_size = settings.max_train_dataset_size
        print('Max train data set: ' + str(max_train_dataset_size))
        if max_train_dataset_size > 0:
            max_train_dataset_size = np.max(max_train_dataset_size, shuffle_size)
            self.train_dataset = self.train_dataset.take(max_train_dataset_size)

        if shuffle_size > 0:
            self.train_dataset = self.train_dataset.shuffle(shuffle_size)
        self.train_dataset = self.train_dataset.repeat()
        if self.settings.use_augmentations:
            self.train_dataset = self.train_dataset.map(lambda color_img, normal_img, voxel, lossmap: (self.augmentation(color_img), normal_img, voxel, lossmap), num_parallel_calls=autotune)
        self.train_dataset = self.train_dataset.map(lambda color_img, normal_img, voxel, lossmap: (self.norm_color_img(color_img), normal_img, voxel, lossmap), num_parallel_calls=autotune)

        self.train_dataset = self.train_dataset.batch(settings.batch_size)
        self.train_dataset = self.train_dataset.prefetch(autotune)

        dynamic_shape = [tf.TensorShape([None, 3, None, None]), tf.TensorShape([None, 3, None, None]),
                             tf.TensorShape([None, settings.amount_of_output_channels, settings.result_size, settings.result_size, settings.result_size])]
        dynamic_shape.append(tf.TensorShape([None, settings.result_size, settings.result_size, settings.result_size]))
        dynamic_shape = tuple(dynamic_shape)
        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, dynamic_shape)
        self.train_init_op = self.iterator.make_initializer(self.train_dataset)
        if self.test_dataset:
            self.test_init_op = self.iterator.make_initializer(self.test_dataset)
        else:
            self.test_init_op = None
        self.validation_init_op = self.iterator.make_initializer(self.validation_dataset)

        self.input_output = self.iterator.get_next()

    def augmentation(self, img):
        color_img = tf.image.random_hue(img, self.settings.hue_delta)
        color_img = tf.image.random_saturation(color_img, self.settings.saturation_lower, self.settings.saturation_upper)
        color_img = tf.image.random_brightness(color_img, self.settings.brightness_delta)
        color_img = tf.image.random_contrast(color_img, self.settings.contrast_lower, self.settings.contrast_upper)
        return color_img

    def _get_dataset_from_path(self, paths, should_shuffle=True):
        if paths:
            dataset_of_file_paths = tf.data.Dataset.from_tensor_slices(paths)
            if should_shuffle:
                dataset_of_file_paths = dataset_of_file_paths.shuffle(len(paths) + 2)
            tfrecord_dataset = dataset_of_file_paths.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), cycle_length=2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return tfrecord_dataset.map(self._deserialize_tfrecord)
        else:
            raise Exception("Something of with the path: {}".format(paths))

    def norm_color_img(self, color_img):
        color_img -= self.mean_img
        color_img /= 150.
        color_img = tf.transpose(color_img, (2, 0, 1))
        return color_img

    def get_input(self):
        return tf.concat([self.input_output[0], self.input_output[1]], axis=1)

    def get_output(self):
        return self.input_output[2]

    def get_loss_map(self):
        if self.settings.end_d_type == tf.float32:
            return self.input_output[3]
        else:
            return tf.cast(self.input_output[3], self.settings.end_d_type)

    def denormalize_input(self, input):
        changed = np.transpose(input, (1, 2, 0))
        img = changed[:, :, :3] * 150 + self.mean_img
        normal_img = changed[:, :, 3:] + self.normal_mean_img
        return np.concatenate([img, normal_img], axis=2)

    def denormalize_output(self, input):
        return np.transpose(input, (1, 2, 3, 0))

    def _deserialize_tfrecord(self, example_proto):
        keys_to_features = {'color': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                            'normal': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                            'lossmap': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                            'voxel': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        color = tf.reshape(parsed_features['color'], (512, 512, 3))
        normal = tf.reshape(parsed_features['normal'], (3, 512, 512))
        lossmap = tf.reshape(parsed_features['lossmap'], (32, 32, 32))
        voxel = tf.reshape(parsed_features['voxel'], (self.amount_of_output_channels, 32, 32, 32))
        parsed_features = None

        return (color, normal, voxel, lossmap)
