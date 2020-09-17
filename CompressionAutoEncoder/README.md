# Compression with an AutoEncoder 

This module encodes the 512³ voxelgrids to 32³x64.

## Changing the config paths

Before, you can use any of these commands please change the following keys in the `config.json` file.

* "train_path": the path to the data used for training 
* "val_path": the path to the data used for the validation (best just move one of the house ids outside to use it as validation sample) 
* "model_save_path": the storage place for the model weights, for that we used a folder named `ae_model` in our data folder. 


## Decoding of the encoded voxelgrids

For decoding a compressed voxelmap, one can use the decode.py script, which requires a trained model. 
You can either use our pretrained model, or train your own model.

For decoding scene just 
```
python decode.py ../data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid/output_0.hdf5
```
This will create the file: `../data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid/output_0_decoded.hdf5` 


## Encoding of voxelgrids

For endcoding all voxelgrids contained in a directory, one can use the encode.py script:

```
python encode.py ../data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid --batch_size 2048
```

This will load the model stored at `../data/ae_model` and use it to compress the voxelgrids contained in all .hdf5 files of all subdirectories in the given path. 
The result will be stored in the same .hdf5 file at the key `encoded_voxelgrid`.

The model used here, is stored in `../data/ae_model/decoder.h5` and `../data/ae_model/encoder.h5`, our pretrained models can be downloaded with this [script](../download_models.py).
If you want to change this path, check the config.json.

Use the biggest batch size possible depending on your GPU memory size (We used 2048 for 12 GB RAM). 

This can be automized again with the [generate_decoded_outputs.py](generate_encoded_outputs.py) script. 
So that, each voxelgrid container file `output_0.hdf5` gets a new key, called `encoded_voxelgrid`, which contains this compressed voxelgrid.

## Training of the model

If the SDFGen was already executed a bunch of house ids with corresponding voxelgrids should be present in the data folder.

By executing the [train.py](train.py), our auto encoder is trained with the generated data in the `data` folder.
The paths are defined in the `config.json` file.

The model weights are saved every 1.000 iterations and after 12.000 iterations the model is completely trained.

## View data

To view these voxelgrids one can use the [TSDFRenderer](../TSDFRenderer).
