# SDFGen

This module generates the TSDF volumes based on the `house.obj` file and some camera poses.

## Build

#### HDF5 

First step is to download and install HDF5, which we use to store the resulting data in a compressed format.

First you have to download this file: 

 [https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/CMake-hdf5-1.10.6.tar.gz](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/CMake-hdf5-1.10.6.tar.gz)

Unzip and execute the `CMake-hdf5-1.10.6/build-unix.sh` file, this will take a few minutes.

After running through without any errors, there will be a file with the name: `HDF5-1.10.6-Linux.tar.gz`.
This file contains the complete build, including the include, lib, bin and share folder.
Unpack this folder to a new location. 
In it you will find this folder: `HDF5-1.10.6-Linux/HDF_Group/HDF5/1.10.6/share/cmake/hdf5`, which has to be set in the `CMakeLists.txt` file as the HDF5_DIR.

#### TCLAP

In the second step you need to install TCLAP.

http://tclap.sourceforge.net/

Download the source files, we used version: `1.2.2`. 
Unzip the downloaded file. And update the path in the `CMakeLists.txt`.

#### Building of SDFGen

After downloading and updating the paths, you only have to build the current project with the given `CMakeLists.txt`.

```shell script
mkdir cmake
cd cmake
cmake -DCMAKE_BUILD_TYPE=RELEASE .. 
make -j 8
```

## Usage

You need a few things to start a `TSDF` generation run.

You need an object file generated (for example generated via the SUNCG folder) and also a cameraposition file also generated via the SUNCGToolBox
After that:6 
```
./sdfgen -o {SUNCG_FOLDER}/house/10704e82d0ef2bf37a658af4fb81c06c/house.obj -c {SUNCG_FOLDER}/house/10704e82d0ef2bf37a658af4fb81c06c/camerapositions -r 128 -f output_folder
```

SDFGen has a lot of tuneable hyperparameters, 

Again, we provide here also a script to do this automatically for the generated `house.obj` and `camerapositions`.

Please, change the paths in [generate_tsdf_volumes.py](generate_tsdf_volumes.py), so that it will convert all generate `house.obj` and `camera positions` into TSDF voxelgrids.


## View data

To view these voxelgrids one can use the [TSDFRenderer](../TSDFRenderer).



