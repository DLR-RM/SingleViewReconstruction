import glob
import multiprocessing
import os
import subprocess
import argparse

import h5py
import tensorflow as tf
import numpy as np

from src.TreeModel import TreeModel
from src.SettingsReader import SettingsReader


def convert_to_uint16(input, threshold=0.1):
    if input.dtype != np.uint16:
        max_value = float(np.iinfo(np.uint16).max) * 0.5
        return (input / threshold * max_value + max_value).astype(np.uint16)
    return input


def denormalize_input(input, mean_img, normal_mean_img):
    changed = np.transpose(input, (1, 2, 0))
    img = changed[:, :, :3] * 150 + mean_img
    normal_img = changed[:, :, 3:] + normal_mean_img
    return np.concatenate([img, normal_img], axis=2)


def denormalize_output(input):
    return np.transpose(input, (1, 2, 3, 0))


class OutputGenerator(object):

    def __init__(self, mean_img, normal_mean_img, folder_path_to_save_to):
        self._mean_img = mean_img
        self._normal_mean_img = normal_mean_img
        self._path_collection = []
        self._folder_path_to_save_to = folder_path_to_save_to
        self.final_paths = []

    def _save_voxel_unencoded(self, output, file_path, image):
        output = denormalize_output(output)
        np.save(file_path, output)
        self._path_collection.append((file_path, image))

    def generate_outputs(self, image, output, name):
        image = denormalize_input(image, self._mean_img, self._normal_mean_img)
        name = name[:name.rfind(".")]
        file_path = os.path.join(self._folder_path_to_save_to, 'output_' + str(name) + ".npy")
        self._save_voxel_unencoded(output, file_path, image)

    def decode_outputs(self):
        print("Start decoding")
        path = os.path.join(self._folder_path_to_save_to, 'output_@.npy')
        decode_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "CompressionAutoEncoder", "decode.py"))
        cmd = "python {} {} --store_as_npy --empty_block_detection_threshold=1e-2".format(decode_script, path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        for path, image in self._path_collection:
            decoded_path = path[:path.find('.')] + '_decoded.npy'
            if not os.path.exists(decoded_path):
                raise Exception("This file was not decoded: {}".format(decoded_path))
            data = np.load(decoded_path)
            data = convert_to_uint16(data)
            os.remove(decoded_path)
            os.remove(path)
            result_path = path.replace(".npy", ".hdf5")
            with h5py.File(result_path, 'w') as output:
                output.create_dataset("voxelgrid", data=data, compression='gzip')
                output.create_dataset("colors", data=image, compression='gzip')
            print("Stored final voxelgrid in {}".format(result_path))
            self.final_paths.append(result_path)


def predict_some_sample_points(image_paths, model_path, folder_path_to_save_to, use_pretrained_weights,
                               use_gen_normals):
    if not os.path.exists(model_path + ".meta"):
        raise Exception("Model does not exist: " + model_path)

    if len(image_paths) == 0:
        raise Exception("The list of .hdf5 containers is empty!")

    folder_path_to_save_to = os.path.abspath(folder_path_to_save_to)

    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    settings_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "settings_file.yml"))
    if os.path.exists(settings_file_path):
        settings = SettingsReader(settings_file_path, data_folder)
    else:
        raise Exception("The settings file could not be found!")

    with h5py.File(os.path.join(data_folder, 'color_normal_mean.hdf5'), 'r') as data:
        mean_img = np.array(data["color"])
        normal_mean_img = np.array(data["normal"])

    settings.batch_size = 1
    settings.shuffle_size = 1
    settings.use_augmentations = False

    used_shape = [None, 3, settings.img_shape[1], settings.img_shape[2]]
    color_input = tf.placeholder(tf.float32, used_shape)
    normal_input = tf.placeholder(tf.float32, used_shape)

    input_to_network = tf.concat([color_input, normal_input], axis=1)

    model = TreeModel(input=input_to_network, settings=settings)
    last_layer = model.last_layer

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def run_tf(i_unused, return_dict):

        def collect_results(sess):
            result = [[], [], []]
            current_config = [input_to_network, last_layer]
            counter = 0
            for image_path in image_paths:
                if not os.path.exists(image_path) or not image_path.endswith(".hdf5"):
                    continue
                with h5py.File(image_path, "r") as file:
                    if "colors" in file.keys():
                        color_o = np.array(file["colors"], dtype=np.float32) - mean_img
                        color_o /= 150.
                        color_o = np.expand_dims(np.transpose(color_o, (2, 0, 1)), axis=0)  # channel first
                    if "normals" in file.keys() or "normal_gen" in file.keys():
                        if not use_gen_normals and "normals" in file.keys():
                            normal_o = np.array(file["normals"], dtype=np.float32)
                        elif use_gen_normals and "normal_gen" in file.keys():
                            normal_o = np.array(file["normal_gen"], dtype=np.float32)
                        else:
                            continue
                        # the original training data, was in sRGB, this has to be recreated
                        if use_pretrained_weights:
                            normal_o = np.power(normal_o, 2.2)
                        normal_o -= normal_mean_img
                        normal_o = np.expand_dims(np.transpose(normal_o, (2, 0, 1)), axis=0)
                    else:
                        continue
                quit_break = False
                res = sess.run(current_config, feed_dict={color_input: color_o, normal_input: normal_o})
                for i in range(2):
                    result[i].append(res[i][0])
                result[2].append(os.path.basename(image_path))
                counter += 1
                print("Done with: {} of {}".format(counter, len(image_paths)))
                if quit_break:
                    break
            return result

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver.restore(sess, model_path)

            results = collect_results(sess)

            return_dict['result'] = (results,)
        print("Session is done")

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=run_tf, args=(0, return_dict))
    p.start()
    p.join()
    results = return_dict['result'][0]
    og = OutputGenerator(mean_img, normal_mean_img, folder_path_to_save_to)
    for i in range(len(results[0])):
        print(i)
        og.generate_outputs(results[0][i], results[1][i], results[2][i])

    og.decode_outputs()

    print('\n' + folder_path_to_save_to)
    return og.final_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This predicts based on an .hdf5 container with a color and normal image "
                                     "a TSDF voxelgrid. The model should be in 'model/model.ckpt'")
    parser.add_argument("path", help="Path to the .hdf5 container, all @ will be replaced with *.")
    parser.add_argument("--output", help="Path to where to save the output files", required=True)
    parser.add_argument("--use_gen_normal", help="Use a generated normal image. Could be generated "
                                                 "with UNetNormalGen.", action="store_true")
    parser.add_argument("--use_pretrained_weights", help="Use the pretrained weight mode.", action="store_true")

    args = parser.parse_args()

    hdf5_paths = glob.glob(args.path.replace("@", "*"))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "model.ckpt"))

    predict_some_sample_points(hdf5_paths, model_path, args.output, args.use_pretrained_weights, args.use_gen_normal)

    print("You can view these files now with the TSDFRenderer.")
