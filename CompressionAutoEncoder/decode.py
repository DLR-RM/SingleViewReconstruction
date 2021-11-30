import argparse
import h5py
import os
import numpy as np
from time import time
import tensorflow as tf

from src.dataset import Dataset
from src.autoencoder import Autoencoder
from src.configreader import ConfigReader

parser = argparse.ArgumentParser(description="Decodes the given compressed voxelgrid with the given model.")
parser.add_argument('data_path', type=str, help="The path to a .npy or .hdf5 file which contains the compressed voxelgrid.")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--empty_block_detection_threshold', type=float, default=1e-5,
                    help="The maximum element-wise absolute difference between a latent encoding "
                         "and the empty/full block encoding to set the block empty/full right away.")
parser.add_argument("--store_as_npy", help="Usually the output is saved as a .hdf5 container, "
                                           "using this will save the output as .npy",action="store_true")
args = parser.parse_args()


config_path = os.path.join(os.path.dirname(__file__), "config.json")

config_obj = ConfigReader(config_path)

dataset = Dataset(config_obj)
dataset.batch_size = args.batch_size

model = Autoencoder(config_obj, dataset)
model.set_iterators(eval_from_placeholder=True)
model_save_path = os.path.join(os.path.dirname(__file__), "..", "data", "ae_model")
model.load(model_save_path)

input_ones = np.ones([1, dataset.input_size(), dataset.input_size(), dataset.input_size(), 1])
full_block_latent = model.encode_from_placeholder(input_ones * -dataset.truncation_threshold)
empty_block_latent = model.encode_from_placeholder(input_ones * dataset.truncation_threshold)

data_iterator = dataset.load_custom_data(args.data_path, fast_inference=True, input_is_latent=True, num_threads=1,
                                         full_block_latent=full_block_latent, empty_block_latent=empty_block_latent,
                                         empty_block_detection_threshold=args.empty_block_detection_threshold)
model.set_iterators(eval_from_latent_iterator=data_iterator)
model.load(model_save_path)

batch_container = np.zeros([dataset.number_of_blocks_per_voxelgrid(), dataset.block_size, dataset.block_size, dataset.block_size, 1])


start = time()
try:
    while True:
        section_start = time()
        output, path = model.decode_not_split_input(batch_container)
        path = path.decode('UTF-8')
        output = output[0, :, :, :, 0]
        print("Finished decoding", time() - section_start, path)

        section_start = time()
        if args.store_as_npy:
            file_path = path[:path.rfind(".")] + "_decoded.npy"
            np.save(file_path, output)
        else:
            file_path = path[:path.rfind(".")] + "_decoded.hdf5"
            with h5py.File(file_path, "w") as file:
                file.create_dataset("voxelgrid", data=output, compression='gzip')
        print("Finished writing output to file", time() - section_start)
except tf.errors.OutOfRangeError:
    pass
print("Took time: {}".format(time() - start))
