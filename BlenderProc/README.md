
# BlenderProc

In order to create realistic looking images we used BlenderProc, which is a procedural pipeline based on the open source project blender.
It makes it possible to easily generate new images with physically plausible renderings.

If you have more detailed questions, we refer to the extensive documentation on how to use [BlenderProc](https://github.com/DLR-RM/BlenderProc).

For this project, we only need to generate the color and normal images:

```shell script
python BlenderProc/run.py config.yaml SingleViewReconstruction/data/10704e82d0ef2bf37a658af4fb81c06c/cam/camerapositions SingleViewReconstruction/data/10704e82d0ef2bf37a658af4fb81c06c/obj/house.obj SingleViewReconstruction/data/10704e82d0ef2bf37a658af4fb81c06c/blenderproc
```

This run.py script takes at first the `config.yaml` and three additional arguments:
1. The camera positions file
2. The house.json file
3. The path, where the resulting normal and color images are stored

For this we also offer a script to automize this, which generates for each generate `house.json` and `camerapositions` file, the matching color and normal images, which are also stored in hdf5 containers.
The used python environment only needs the package `pyyaml`.

```shell script
python generate_color_and_normals_imgs.py
```