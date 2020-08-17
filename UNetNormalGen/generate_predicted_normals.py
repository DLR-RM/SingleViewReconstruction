import os
import glob
import argparse

import numpy as np
import h5py
import tensorflow as tf

from source.settings_reader import SettingsReader
from source.model import Model

parser = argparse.ArgumentParser("Predicting normal images for color images given a trained model.")
parser.add_argument("--model_path", help="The path to he model.ckpt file", required=True)
parser.add_argument("--path", help="The path, where to search for the data. The symbol @ will be replaced with a *.", required=True)
parser.add_argument("--use_pretrained", help="Use this, if the pretrained model was used.", action='store_true', default=False)

args = parser.parse_args()

image_paths = []
if os.path.basename(args.path) == "data" and "@" not in args.path:
    image_paths = glob.glob(os.path.join(args.path, "*", "blenderproc", "*.hdf5"))
else:
    image_paths = glob.glob(args.path.replace("@", "*"))

if len(image_paths) == 0:
    raise Exception("No .hdf5 files where found here: {}".format(args.path))


settings_file_path = os.path.join(os.path.dirname(__file__), "settings", "settings_file.yml")
settings = SettingsReader(settings_file_path)

input_ph = tf.placeholder(tf.float32, (None, settings.img_size, settings.img_size, 3))

# create the model
model = Model()
model_result = model.create(input_ph)
last_layer, _, _, _ = model.get_results()

# Saver
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, args.model_path)

    for image_path in image_paths:
        if image_path.endswith(".hdf5"):
            color_img = None
            with h5py.File(image_path, "r") as file:
                if "colors" in file.keys():
                    color_img = np.expand_dims(np.array(file["colors"], dtype=np.float32)[:, :, :3], axis=0) / 255.0
                elif "colors" not in file.keys():
                    print("The colors key is missing in this .hdf5 container: {}".format(image_path))
            if color_img is not None:
                with h5py.File(image_path, "a") as file:
                    normal_gen = sess.run(last_layer, feed_dict={input_ph: color_img})
                    if args.use_pretrained:
                        # pretrained data was in sRGB format, must be corrected here:
                        normal_gen = np.power((normal_gen + 1) / 2.0, 1.0/2.2)
                    if "normal_gen" in file:
                        del file["normal_gen"]
                    file.create_dataset("normal_gen", data=normal_gen[0], compression="gzip")
            print("Done with image path: {}".format(image_path))

print("Done with converting image paths!")




