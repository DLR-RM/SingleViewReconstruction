import glob
import os
from shutil import copyfile
from socket import gethostname
from time import strftime

import numpy as np
import tensorflow as tf
import yaml


class SettingsReader(object):

    def __init__(self, settings_file_path, special_folder_path=None):
        with open(settings_file_path, 'r') as stream:
            settings = yaml.load(stream)

        self.d_type = self.get_tf_type(settings["type"]["start"])
        self.end_d_type = self.get_tf_type(settings["type"]["end"])

        img_size = settings["image"]["size"]
        img_channel = None
        self.img_shape = (img_channel, img_size, img_size)

        self.result_size = settings["result"]["size"]
        self.bootstrap_ratio = settings['bootstrap_ratio']
        self.amount_of_output_channels = settings["result"]["amount_of_output_channels"]

        self.result_shape = (None, self.amount_of_output_channels, self.result_size, self.result_size, self.result_size)

        self.height = settings["tree"]["height"]

        self.use_reflective_padding_3D = self.get_settings_bool(settings["tree"]["use_reflective_padding_3D"])
        self.amount_of_filters_in_first_3D = int(settings["tree"]["amount_of_filters_in_first_3D"])
        self.inner_tree_loss_weight = float(settings['tree']['inner_tree_loss_weight'])
        self.loss_height_weight = self.get_settings_list(settings["tree"]["loss_height_weight"], type=float)

        self.batch_size = settings["batch_size"]
        self.test_batch_size = settings["batch_size"]
        self.test_shuffle_paths = True

        self.test_dataset_size = settings["TestDataSet"]["size"]

        self.filters_for_level = self.get_settings_list(settings["tree"]["filters_for_level"])
        self.residual_levels = self.get_settings_list(settings["tree"]["residual_levels"])
        self.pool_levels = self.get_settings_list(settings["tree"]["pool_levels"])
        self.input_structure = self.get_settings_list(settings["tree"]["input_structure"])

        # 3d structure
        self.dil_values_for_3d = self.get_settings_list(settings["3d_layers"]["dil_values"])
        self.filter_amounts_for_3d = self.get_settings_list(settings["3d_layers"]["filter_amounts"])
        self.layer_structure_for_3d = self.get_settings_list(settings["3d_layers"]["structure"])
        self.dil_values_for_3d_separable = self.get_settings_list(settings["3d_layers"]["dil_values_for_separable"])
        self.filter_amounts_for_3d_separable = self.get_settings_list(settings["3d_layers"]["filter_amount_for_separable"])
        self.use_plane_for_separable_3d = self.get_settings_bool(settings["3d_layers"]["use_plane_for_separable"])
        self.before_3D_loss_weight = float(settings["3d_layers"]["before_3D_loss_weight"])


        self.learning_rate = float(settings["learning_rate"])
        self.filters_in_deepest_node = int(settings["tree"]["filters_in_deepest_node"])

        self.shuffle_size = int(settings['shuffle_size'])
        self.max_train_dataset_size = int(settings['max_train_dataset_size'])


        augmentation = settings["augmentations"]

        self.use_augmentations = self.get_settings_bool(augmentation["use_them"])
        self.brightness_delta = augmentation["brightness_delta"]
        self.contrast_lower = augmentation["contrast_lower"]
        self.contrast_upper = augmentation["contrast_upper"]
        self.hue_delta = augmentation["hue_delta"]
        self.saturation_lower = augmentation["saturation_lower"]
        self.saturation_upper = augmentation["saturation_upper"]
        self.amount_of_threads_used = augmentation['amount_of_threads_used']
        self.use_loss_map = self.get_settings_bool(augmentation['use_loss_map'])

        self.regularizer_scale = settings['regularizer_scale']

        data = settings["data"]
        self.folder_path = data["folder_path"]
        self.data_file_name = data['file_name']
        self.data_file_paths = glob.glob(os.path.join(self.folder_path, self.data_file_name))
        if not self.data_file_paths:
            print("Warning: No data files have been found, the training will not work in this mode")
        else:
            self.data_file_paths.sort()

        self.LOG_DIR = settings['log_dir']
        if '${time_str}' in self.LOG_DIR:
            # get current time to set used folder path
            timestr = strftime("%Y%m%d-%H%M%S")
            self.LOG_DIR = self.LOG_DIR.replace('${time_str}', timestr)

    def get_tf_type(self, input):
        if "float" in input:
            if "32" in input:
                return tf.float32
            elif "16" in input:
                return tf.float16
            else:
                print("No known specific float type: " + str(input))
        elif "uint" in input:
            if "32" in input:
                return tf.uint32
            elif "16" in input:
                return tf.uint16
            elif "8" in input:
                return tf.uint8
            else:
                print("No known specific uint type: " + str(input))
        elif "int" in input:
            if "32" in input:
                return tf.int32
            elif "16" in input:
                return tf.int16
            elif "8" in input:
                return tf.int8
            else:
                print("No known specific int type: " + str(input))
        else:
            print("No known specific type at all: " + str(input))

    def get_settings_bool(self, line):
        if isinstance(line, bool):
            return line
        else:
            return line == "True" or line == "true" or line == "1"

    def get_settings_list(self, line, type=int):
        if isinstance(line, list):
            return list
        if len(line) != 0:
            if ',' in line:
                return [type(ele) for ele in line.split(',')]
            else:
                return [type(line)]
        else:
            return []
