import argparse
import h5py
import os
import numpy as np
from time import time
import tensorflow as tf

from src.dataset import Dataset
from src.autoencoder import Autoencoder
from src.configreader import ConfigReader

parser = argparse.ArgumentParser(description="Encodes the voxelgrids of all .hdf5 files contained in the given path with the given model. Uses the tensor with key 'voxelgrid' as input and stores the output at key'encoded_voxelgrid'")
parser.add_argument('path', type=str, help="The path where to look for subdirectories with .h5py files.")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--threads', type=int, default=4, help="Number of threads to use in the input pipeline.")
args = parser.parse_args()


config_path = os.path.join(os.path.dirname(__file__), "config.json")

config_obj = ConfigReader(config_path)

dataset = Dataset(config_obj)
dataset.batch_size = args.batch_size
data_iterator = dataset.load_custom_data(args.path, fast_inference=True, num_threads=args.threads)


model = Autoencoder(config_obj, dataset)

model.set_iterators(eval_from_input_iterator=data_iterator, eval_from_placeholder=True, eval_uses_fast_inference=True)

model.load(config_obj.data.get_string("model_save_path"))
model.summary()

input_ones = np.ones([1, dataset.input_size(), dataset.input_size(), dataset.input_size(), 1])
full_block_latent = model.encode_from_placeholder(input_ones * -dataset.truncation_threshold)
empty_block_latent = model.encode_from_placeholder(input_ones * dataset.truncation_threshold)

batch_container = np.zeros([dataset.number_of_blocks_per_voxelgrid(), 1, 1, 1, dataset.latent_channel_size])

try:

    while True:
        # Run model
        start = time()
        output, sample_path = model.encode_not_split_input(batch_container, empty_block_latent, full_block_latent)
        sample_path = sample_path.decode('UTF-8')

        print("Compressed shape: " + str(output.shape) + " (" + str(time() - start) + "s) - " + str(sample_path))

        # Store output back into hd5f file (only if non-existing or overwriting is true)
        with h5py.File(sample_path, 'a') as f:
            voxelgrid = output[0]

            if "encoded_voxelgrid" in f.keys():
                if f["encoded_voxelgrid"].shape == output[0].shape:
                    f["encoded_voxelgrid"][...] = voxelgrid
                else:
                    del f["encoded_voxelgrid"]
                    f["encoded_voxelgrid"] = voxelgrid
            else:
                f.create_dataset("encoded_voxelgrid", data=voxelgrid, compression="grip")
except tf.errors.OutOfRangeError:
    pass

