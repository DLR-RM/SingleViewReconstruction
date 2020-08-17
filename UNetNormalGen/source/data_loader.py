
import tensorflow as tf
import glob
import os

autotune = tf.data.experimental.AUTOTUNE

class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, settings):
        super(DataLoader, self).__init__()
        self._batch_size = settings.batch_size
        self._path = settings.data_path
        self._shuffle_size = settings.shuffle_size
        self._dataset_size = settings.max_dataset_size
        self._validation_ratio = settings.validation_ratio
        self._augment = settings.augment
        self._datasets = {}
        self.load_dataset()

    def augmentation(self, img, hue_delta= 0.12, saturation_lower=0.44, saturation_upper=1.6, brightness_delta=0.1,
                     contrast_lower=0.5, contrast_upper=1.5):
        color_img = tf.image.random_hue(img, hue_delta)
        color_img = tf.image.random_saturation(color_img, saturation_lower, saturation_upper)
        color_img = tf.image.random_brightness(color_img, brightness_delta)
        color_img = tf.image.random_contrast(color_img, contrast_lower, contrast_upper)
        return color_img


    # deserialize the td records
    def deserialize_tfrecord(self, example_proto):
        # for each example we define the used features
        keys_to_features = {}
        keys_to_features['colors'] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
        keys_to_features['normals'] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = tf.reshape(parsed_features['colors'], (512, 512, 3))
        label = tf.reshape(parsed_features['normals'], (512, 512, 3))

        # return all of them
        return (img, label)

    def load_dataset(self):

        # if (compressed):
        # uncompress tfrecord
        tfrecords_list = glob.glob(os.path.join(self._path, '*.tfrecord'))
        if len(tfrecords_list) == 0:
            raise Exception("There are no training/validation tf records in this path: {}".format(self._path))

        # read the tf record file
        tfrecord_dataset = tf.data.TFRecordDataset(tfrecords_list, compression_type='GZIP')

        dataset = tfrecord_dataset.map(self.deserialize_tfrecord, num_parallel_calls=autotune)

        val_size = int(self._dataset_size*self._validation_ratio)
        if val_size <= 0:
            raise Exception("The validation size must be bigger than zero, "
                            "increase the dataset size or the validation ratio!")

        # Split train and val
        val_dataset = dataset.take(val_size)
        trn_dataset = dataset.skip(val_size)

        if self._augment:
            # Augment the train dataset
            trn_dataset = trn_dataset.map(lambda color_img, normal_img: (self.augmentation(color_img), normal_img),
                                          num_parallel_calls=autotune)

        # Repeat
        trn_dataset = trn_dataset.repeat()
        val_dataset = val_dataset.repeat()

        # Shuffle the dataset
        trn_dataset = trn_dataset.shuffle(self._shuffle_size)

        # Batch
        trn_dataset = trn_dataset.batch(self._batch_size)
        val_dataset = val_dataset.batch(self._batch_size)

        self._trn_dataset = trn_dataset.prefetch(buffer_size=autotune)
        self._val_dataset = val_dataset.prefetch(buffer_size=autotune)

        # return self.trn_dataset, self.val_dataset

    def load_default_iterator(self):
        self.trn_iter = self._trn_dataset.make_initializable_iterator()
        self.val_iter = self._val_dataset.make_initializable_iterator()

        return self.trn_iter, self.val_iter

