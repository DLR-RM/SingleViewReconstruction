from itertools import product
import h5py
import glob
import os
import random

import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, config):
        """
        Args:
            config: The configuration config.
        """
        self.config = config.data

        self.data_size = self.config.get_int("data_size")
        self.truncation_threshold = self.config.get_float("truncation_threshold")

        self.batch_size = self.config.get_int("batch_size")
        self.eval_batch_size = self.config.get_int("eval_batch_size")
        if self.eval_batch_size is 0:
            self.eval_batch_size = self.batch_size

        self.additional_border = self.config.get_int("additional_border")
        self.additional_blocks = self.config.get_int("additional_blocks")
        self.focused_blocks = self.config.get_int("focused_blocks")
        self.block_size = self.config.get_int("block_size")
        self.train_path = self.config.get_string("train_path")
        self.val_path = self.config.get_string("val_path")

        self.train_hard_samples_path = self.config.get_string("train_hard_samples_path")
        self.val_hard_samples_path = self.config.get_string("val_hard_samples_path")
        self.latent_channel_size = 64

        self.complete_latent_mask = np.ones([self.data_size // self.block_size] * 3 + [1], dtype=np.float32)
        self.complete_latent_mask = self.add_padding_to_latent(self.complete_latent_mask)

    def total_blocks(self, focused_blocks=None):
        """ Computes the number of blocks per axis which constitute the input. """
        return self.additional_blocks * 2 + (self.focused_blocks if focused_blocks is None else focused_blocks)

    def input_size(self, focused_blocks=None):
        """ Computes the total side length of the input. """
        return self.block_size * self.total_blocks(focused_blocks) + 2 * self.additional_border

    def output_size(self, focused_blocks=None):
        """ Computes the total side length of the input. """
        return self.block_size * self.total_blocks(focused_blocks)

    def focus_size(self, focused_blocks=None):
        """ Computes the side length of the focused input. """
        return (self.focused_blocks if focused_blocks is None else focused_blocks) * self.block_size

    def input_padding(self):
        """ Computes the total padding around the focused input. """
        return self.additional_border + self.block_size * self.additional_blocks

    def _truncate(self, data, latent_masks=None):
        """ Clips all values at [-truncation_threshold, truncation_threshold] """
        if self.truncation_threshold > 0:
            data = tf.clip_by_value(data, -self.truncation_threshold, self.truncation_threshold)

        if latent_masks is not None:
            return data, latent_masks
        else:
            return data

    def _filter_truncated(self, data, latent_masks=None):
        """ Removes all block which only consists of clipped values (ignoring additional border around input) """
        if self.truncation_threshold > 0:
            if latent_masks is not None:
                total_blocks = (self.additional_blocks * 2 + self.focused_blocks) * self.block_size
                # Cut out the data without the additional border.
                data = data[self.additional_border:total_blocks, self.additional_border:total_blocks, self.additional_border:total_blocks, :]

            pred = tf.math.less(tf.random.uniform([1], 0, 1)[0], 0.99)

            def f1():
                return tf.logical_not(tf.reduce_all(tf.equal(tf.abs(data), self.truncation_threshold)))

            def f2():
                return True

            return tf.cond(pred, true_fn=f1, false_fn=f2)
        else:
            return True

    def _add_cropped_label(self, data, latent_masks):
        """ Cuts out the focused blocks and returns them as labels. """
        padding = self.input_padding()
        return data, data[padding:padding + self.block_size * self.focused_blocks, padding:padding + self.block_size * self.focused_blocks, padding:padding + self.block_size * self.focused_blocks, :], latent_masks

    def _add_label(self, data):
        """ Adds the input also as label """
        return data, data

    def possible_input_window_positions(self, focused_blocks=None):
        """ Returns all possible possible positions of the window on the input data considering padding and stride. """
        return np.array(list(product(range(0, self.data_size // (self.block_size * (self.focused_blocks if focused_blocks is None else focused_blocks))), repeat=3)))

    def input_window_slice(self, window_pos, focused_blocks=None):
        """ Builds the slice for the given window position on the input data. """
        start = window_pos * self.focus_size(focused_blocks)
        return [slice(start[i], start[i] + self.input_size(focused_blocks)) for i in range(3)] + [slice(None, None)]

    def latent_mask_window_slice(self, window_pos, focused_blocks=None):
        """ Builds the slice for the given window position on the latent mask. """
        start = window_pos * (self.focused_blocks if focused_blocks is None else focused_blocks)
        return [slice(start[i], start[i] + self.total_blocks(focused_blocks)) for i in range(3)] + [slice(None, None)]

    def output_window_slice(self, window_pos, focused_blocks=None):
        """ Builds the slice for the given window position on the output data. """
        start = window_pos * self.focus_size(focused_blocks)
        return [slice(start[i], start[i] + self.focus_size(focused_blocks)) for i in range(3)] + [slice(None, None)]

    def add_padding_to_input(self, input):
        """ Adds the required padding (for encoder+decoder) around the input voxel grid. """
        padding = self.input_padding()
        return np.pad(input, ([[0, 0]] if len(input.shape) > 4 else []) + [[padding, padding], [padding, padding], [padding, padding], [0, 0]], 'constant')

    def add_padding_to_latent(self, input):
        """ Adds the required padding (for decoder) around the latent respresentation of a voxel grid. """
        return np.pad(input, ([[0, 0]] if len(input.shape) > 4 else []) + [[self.additional_blocks, self.additional_blocks], [self.additional_blocks, self.additional_blocks], [self.additional_blocks, self.additional_blocks], [0, 0]], 'constant')

    def number_of_blocks_per_voxelgrid(self):
        return len(self.possible_input_window_positions())

    def h5py_reader(self, path, shuffle, input_is_latent, is_inference):
        """ Generator method for reading data out of all found h5py files.

        If do_split is true, the generator will slide over each voxel grid and yield
        the content of the current window. This will

        Args:
            path: The path where to look for subdirectories with .h5py files
            do_split: True, if the input should be split up already.
            shuffle: True, if the list of h5py files should be shuffled before reading them in.

        Returns:
            Numpy array(s)
        """
        # Read in filenames
        path = path.decode('UTF-8')
        path = path.replace("@", "*")
        paths = glob.glob(os.path.join(path, "*.hdf5") if not path.endswith('.hdf5') else path)
        paths = [path for path in paths if "_loss_avg.hdf5" not in path]
        if not paths:
            raise Exception("No hdf5 files were found in path: {}".format(path))
        if shuffle:
            random.shuffle(paths)
        else:
            print("Sort the {} paths!".format(len(paths)))
            paths.sort()

        for sample_path in paths:
            data = self._read_h5py(sample_path, input_is_latent)
            if data is None:
                continue
            if not is_inference:
                yield (data,)
            else:
                yield (data, sample_path)

    def _read_h5py(self, sample_path, input_is_latent):
        # Read voxel map
        with h5py.File(sample_path, 'r') as f:
            used_key = "encoded_voxelgrid" if input_is_latent else "voxelgrid"
            if used_key not in f:
                return None
            full_input = np.array(f[used_key]).astype(np.float)
            if not input_is_latent:
                # Add channel dimension, s.t. the network can handle the input.
                full_input = np.expand_dims(full_input, -1)

        return full_input

    def npy_path_prepare(self, path, shuffle):
        if path.endswith('.npy') and '@' in path:
            path = path.replace('@', '*')
            tmp_paths = glob.glob(path)
            paths = []
            for path in tmp_paths:
                if not 'decoded' in path:
                    paths.append(path)
        else:
            paths = glob.glob(os.path.join(path, "*/*.npy") if not path.endswith('.npy') else path)
        if shuffle:
            random.shuffle(paths)
        else:
            paths.sort()
        return paths

    def npy_reader(self, path, shuffle, is_inference):
        path = path.decode('UTF-8')
        paths = self.npy_path_prepare(path, shuffle)
        # Read in filenames
        for sample_path in paths:
            # Read voxel map
            voxelgrid = np.load(sample_path)

            if not is_inference:
                yield (voxelgrid,)
            else:
                yield (voxelgrid, sample_path)

    def decode(self, input_window, latent_mask_window=None):
        # Decode values (map [0, uint16.max] to [-trunc_threshold, trunc_threshold])
        input_window = input_window / np.iinfo(np.uint16).max * self.truncation_threshold * 2 - self.truncation_threshold

        if latent_mask_window is not None:
            return input_window, latent_mask_window
        else:
            return input_window

    def _decode_tf_records(self, serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'data': tf.FixedLenFeature([self.data_size ** 3], tf.float32)
        })

        data = features['data']
        data = tf.reshape(data, [self.data_size] * 3 + [1])

        padding = self.input_padding()
        data = tf.pad(data, [[padding, padding], [padding, padding], [padding, padding], [0, 0]], "SYMMETRIC")

        return data

    def _pad(self, data, path=None, input_is_latent=False):
        padding = self.input_padding() if not input_is_latent else self.additional_blocks
        data = tf.pad(data, [[padding, padding], [padding, padding], [padding, padding], [0, 0]], "SYMMETRIC")

        if path is None:
            return data
        else:
            return data, path

    def _decode_hard_samples_tf_records(self, serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'input_data': tf.FixedLenFeature([22 ** 3], tf.float32),
            'label_data': tf.FixedLenFeature([8 ** 3], tf.float32),
            'latent_mask_data': tf.FixedLenFeature([1 ** 3], tf.float32)
        })

        input = features['input_data']
        input = tf.reshape(input, [22] * 3 + [1])

        label = features['label_data']
        label = tf.reshape(label, [8] * 3 + [1])

        latent_mask = features['latent_mask_data']
        latent_mask = tf.reshape(latent_mask, [1] * 3 + [1])

        return input, label, latent_mask

    def _random_flip(self, input_window, latent_mask_window=None):
        axis = tf.random.uniform([1], -1, 3, dtype=tf.dtypes.int32)
        input_window = tf.cond(tf.math.less(axis[0], 0), true_fn=lambda: input_window, false_fn=lambda: tf.reverse(input_window, axis))
        latent_mask_window = tf.cond(tf.math.less(axis[0], 0), true_fn=lambda: latent_mask_window, false_fn=lambda: tf.reverse(latent_mask_window, axis))

        if latent_mask_window is not None:
            return input_window, latent_mask_window
        else:
            return input_window

    def _split(self, input, path=None, input_is_latent=False, filter_clipped_blocks=False, full_block_latent=None, empty_block_latent=None, empty_block_detection_threshold=1e-5):
        input = tf.expand_dims(input, 0)
        window_size = self.input_size() if not input_is_latent else self.total_blocks()
        strides = self.focus_size() if not input_is_latent else self.focused_blocks
        input_patches = tf.extract_volume_patches(input, [1, window_size, window_size, window_size, 1], [1, strides, strides, strides, 1], "VALID")

        complete_latent_mask = tf.constant(np.expand_dims(self.complete_latent_mask, 0))
        latent_patches = tf.extract_volume_patches(complete_latent_mask, [1, self.total_blocks(), self.total_blocks(), self.total_blocks(), 1], [1, self.focused_blocks, self.focused_blocks, self.focused_blocks, 1], "VALID")

        input_block_size = self.input_size() if not input_is_latent else self.total_blocks()
        num_blocks = self.number_of_blocks_per_voxelgrid()
        input_patches = tf.reshape(input_patches, [num_blocks, input_block_size, input_block_size, input_block_size, 1 if not input_is_latent else self.latent_channel_size])
        latent_patches = tf.reshape(latent_patches, [num_blocks, self.total_blocks(), self.total_blocks(), self.total_blocks(), 1])

        if filter_clipped_blocks:
            # Create masks for empty and full blocks
            if empty_block_latent is None:
                empty_blocks = tf.reduce_all(tf.equal(input_patches, self.truncation_threshold), [1, 2, 3, 4])
            else:
                empty_blocks = tf.reduce_all(tf.less_equal(tf.abs(input_patches - empty_block_latent), empty_block_detection_threshold), [1, 2, 3, 4])
            if full_block_latent is None:
                filled_blocks = tf.reduce_all(tf.equal(input_patches, -self.truncation_threshold), [1, 2, 3, 4])
            else:
                filled_blocks = tf.reduce_all(tf.less_equal(tf.abs(input_patches - full_block_latent), empty_block_detection_threshold), [1, 2, 3, 4])

            # Create an array which contains 1 for empty, -1 for filled and 0 for all other blocks
            ones = tf.ones((num_blocks,))
            types = tf.where(empty_blocks, ones, tf.where(filled_blocks, ones * -1, ones * 0))
            types = tf.expand_dims(types, 0)

            # Remove all blocks which are filled or empty
            indices_to_keep = tf.logical_not(tf.logical_or(empty_blocks, filled_blocks))
            input_patches = tf.boolean_mask(input_patches, indices_to_keep)
            latent_patches = tf.boolean_mask(latent_patches, indices_to_keep)

            return input_patches, latent_patches, types, path

        else:
            return input_patches, latent_patches

    def build_filtered_dataset(self, inputs, latent, types, path, batch_size):
        # Group all blocks into batches (in contrast to using .batch() in this way there are no batches containing blocks from two different scenes)
        input_dataset = tf.data.Dataset.from_tensor_slices(inputs).window(batch_size).flat_map(lambda window: window.batch(batch_size))
        latent_dataset = tf.data.Dataset.from_tensor_slices(latent).window(batch_size).flat_map(lambda window: window.batch(batch_size))

        # Create the first sample which contains the type array (0, -1, 1) and the sample path
        empty_paths = tf.constant([""], tf.string)
        empty_paths = tf.tile(empty_paths, tf.expand_dims(tf.cast(tf.ceil(tf.cast(tf.shape(inputs)[0], tf.float32) / float(batch_size)), tf.int32), 0))
        empty_paths_dataset = tf.data.Dataset.from_tensor_slices(empty_paths)
        first_sample_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([1, 1, self.data_size, self.data_size, self.data_size, 1]), types, tf.expand_dims(path, 0)))

        # Concat both
        return_dataset = first_sample_dataset.concatenate(tf.data.Dataset.zip((input_dataset, latent_dataset, empty_paths_dataset)))

        return return_dataset

    def _build_data_reader(self, path, shuffle, input_is_latent, num_threads, is_inference):
        input_size = ([self.data_size] * 3 + [1],)
        latent_size = ([self.data_size // self.block_size] * 3 + [self.latent_channel_size],)
        output_shape = latent_size if input_is_latent else input_size
        if path.endswith('.npy'):
            dataset = tf.data.Dataset.from_generator(self.npy_reader, (tf.float32, tf.string) if is_inference else (tf.float32,), args=(path, shuffle, is_inference), output_shapes=(output_shape[0], []) if is_inference else output_shape)
        else:
            dataset = tf.data.Dataset.from_generator(self.h5py_reader, (tf.float32, tf.string) if is_inference else (tf.float32,), args=(path, shuffle, input_is_latent, is_inference), output_shapes=(output_shape[0], []) if is_inference else output_shape)

        dataset = dataset.map(lambda x, y=None: self._pad(x, y, input_is_latent=input_is_latent), num_parallel_calls=num_threads)

        if not input_is_latent:
            dataset = dataset.map(self.decode, num_parallel_calls=num_threads)

        return dataset

    def load_data(self, path, batch_size, do_split=True, shuffle=True, repeat=True, filter_truncated=True, do_dataaugmentation=False, drop_remainder=False, hard_samples_path=None, input_is_latent=False, num_threads=None):
        """ Creates an iterator for data inside the given path.

        Args:
            path: The path where to look for subdirectories with .h5py files
            batch_size: int
            do_split: True, if the input should be split up already.
            shuffle: True, if the list of h5py files should be shuffled before reading them in and if the input blocks should be shuffled after extracting them.
            repeat: True, if the iterator should run endlessly.

        Returns:
            The iterator
        """
        # Read in files
        dataset = self._build_data_reader(path, shuffle, input_is_latent, num_threads, False)

        if do_split:
            dataset = dataset.map(lambda x: self._split(x, input_is_latent=input_is_latent), num_parallel_calls=num_threads)
            dataset = dataset.prefetch(1)
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))))

        if not input_is_latent:
            # Clip at threshold
            dataset = dataset.map(self._truncate, num_parallel_calls=num_threads)

        if filter_truncated and not input_is_latent:
            # Throw away inputs which just consist of clipped values.
            dataset = dataset.filter(lambda x, y: self._filter_truncated(x, y))

        if do_dataaugmentation and self.config.get_bool("data_augmentation"):
            dataset = dataset.map(self._random_flip, num_parallel_calls=num_threads)

        # Add label
        if do_split:
            dataset = dataset.map(self._add_cropped_label, num_parallel_calls=num_threads)
        else:
            dataset = dataset.map(self._add_label, num_parallel_calls=num_threads)

        if hard_samples_path is not None and len(hard_samples_path) > 0:
            hard_samples_dataset = tf.data.TFRecordDataset(hard_samples_path)
            hard_samples_dataset = hard_samples_dataset.map(self._decode_hard_samples_tf_records, num_parallel_calls=num_threads)
            dataset = tf.data.Dataset.range(2).interleave(lambda x: dataset if x is 0 else hard_samples_dataset, cycle_length=2, block_length=1)

        # Shuffle, batch and repeat
        if shuffle:
            dataset = dataset.shuffle(1000 + batch_size * 3)

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if repeat:
            dataset = dataset.repeat(None)
        dataset = dataset.prefetch(1)

        return dataset.make_one_shot_iterator()

    def load_data_for_fast_inference(self, path, batch_size, input_is_latent=False, num_threads=4, full_block_latent=None, empty_block_latent=None, empty_block_detection_threshold=1e-5):
        # Read in files
        dataset = self._build_data_reader(path, False, input_is_latent, num_threads, True)

        dataset = dataset.map(lambda x, y: self._split(x, y, input_is_latent=input_is_latent, filter_clipped_blocks=True, full_block_latent=full_block_latent, empty_block_latent=empty_block_latent, empty_block_detection_threshold=empty_block_detection_threshold), num_parallel_calls=num_threads)
        dataset = dataset.prefetch(2)
        dataset = dataset.flat_map(lambda x, y, z, w: self.build_filtered_dataset(x, y, z, w, batch_size))

        dataset = dataset.prefetch(1)

        return dataset.make_one_shot_iterator()

    def load_train_data(self, repeat=True):
        """ Returns an iterator which yields training data in block size """
        return self.load_data(self.train_path, self.batch_size, do_dataaugmentation=True, drop_remainder=True, repeat=repeat, hard_samples_path=self.train_hard_samples_path)

    def load_val_data(self, repeat=True):
        """ Returns an iterator which yields validation data in block size """
        return self.load_data(self.val_path, self.batch_size, drop_remainder=True, repeat=repeat, hard_samples_path=self.val_hard_samples_path)

    def load_eval_data(self):
        """ Returns an iterator which yields validation data as whole voxel grids """
        return self.load_data(self.val_path, self.eval_batch_size, True, False, True, False)

    def load_custom_data(self, path, input_is_latent=False, fast_inference=False, num_threads=4, full_block_latent=None, empty_block_latent=None, empty_block_detection_threshold=1e-5):
        """ Returns an iterator which yields the data found at the given path as whole voxel grids """
        if not fast_inference:
            return self.load_data(path, self.batch_size, True, False, False, False, input_is_latent=input_is_latent, num_threads=num_threads)
        else:
            return self.load_data_for_fast_inference(path, self.batch_size, input_is_latent, num_threads, full_block_latent, empty_block_latent, empty_block_detection_threshold)


