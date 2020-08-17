# SUNCG data generation

We have used SUNCG for the data generation, be aware that SUNCG is no longer publicly available. 
So you can only progress here, if you already have a downloaded version of SUNCG.
Be aware that, we are not at liberty to release the assets ourselves.

Assuming you have this, we use the SUNCG Toolbox: [https://github.com/tinytangent/SUNCGtoolbox](https://github.com/tinytangent/SUNCGtoolbox)

So, please follow their instructions on how to download and build it.

Afterwards, you can convert some house.json files, which you can find in the house folder. 
Where each hash id in this folder is a certain house.

```shell script
cd house/10704e82d0ef2bf37a658af4fb81c06c
{INSTALL_FOLDER}/SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json house.obj
{INSTALL_FOLDER}/SUNCGtoolbox/gaps/bin/x86_64/scn2cam house.json camerapositions
```

You will need the house.obj file and the camera positions for the [SDFGen](../SDFGen) and also for the generation of the color & normal images.

This process has to be done for all `house.json` files, which you want to use for the training or testing.

We offer here a script to do this for you automatically, see [here](generate_house_objs_and_cameras.py)

Please change these two paths: 

```python
suncg_toolbox_bin_path = "/home/max/workspace/SUNCGtoolbox/gaps/bin/x86_64"
suncg_house_folder = "/home/max/Downloads/version_1.1.0/house"
```

The house.obj and camera position files are then stored in a [data](data) folder, which is automatically generated.
With the following structure:
``` 
data 
    feeb4209a2f65807d3a2288b85f44a0e <-- this is the house id
        obj 
            house.obj <-- the generated house.obj file
        cam
            camerapositions <-- the generated camerapositions file
```

