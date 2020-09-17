# U-Net for the normal generation

In this module we generate normals based on a single color image, those are used as an input to the SingleViewReconstruction Network.

### Data generation

In order to generate the data to train such a model you can use the script [data/generate_tfrecords.py](data/generate_tfrecords.py).

Be aware you first need to generate some images with [BlenderProc](../BlenderProc). 

```shell script
python data/generate_tfrecords.py --path ../data/@/blenderproc --out data/
```

This will generate `tfrecord` files in the `UNetNormalGen/data` folder, by using the generate images from BlenderProc.

### Training

After the generation of the data you can train a new model.

With the [train.py](train.py) script, it trains a new model and stores the resulting model in the `logs` folder.

### Prediction of the generated normals

For the prediction, you need a pretrained model, you can either train one yourself or use our pretrained model.

First download the model with this [script](../download_models.py) and move the unzipped files in a new folder in `UNetNormalGen/model`.

Now run the prediction pipeline:

```shell script
python generate_predicted_normals.py --model_path model/model.ckpt --path ../data
```

If the `data` folder is provided, where [BlenderProc](../BlenderProc) already generated some color images, it automatically adds to these `.hdf5` containers a new data block with the key `"normal_gen"`.
This will contain the generated normal image, corresponding to the color image.

This can also be used with a bunch of `.hdf5` files like: 

```shell script
python generate_predicted_normals.py --model_path model/model.ckpt --path ../data/@/blenderproc/@.hdf5
```

Please use @ instead of the *, it will be replaced internally.

