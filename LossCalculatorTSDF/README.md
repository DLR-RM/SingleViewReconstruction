# Loss Calculator TSDF 

The LossCalculator generates the weights used for the loss shaping as described in the paper.


## Build

#### HDF5 
As you have build SDFGen before, you can now reuse the `HDF5` package. 
For that, you need to set the HDF5_DIR in the `CMakeLists.txt` file as before.

#### TCLAP

Update the path in the `CMakeLists.txt`, the same as in SDFGen.

#### Building of LossCalculator 

After updating the paths, you only have to build the current project with the given `CMakeLists.txt`.

```shell script
mkdir cmake
cd cmake
cmake -DCMAKE_BUILD_TYPE=RELEASE .. 
make -j 8
```


## Usage

You can now use the LossCalculator, which takes three arguments, the first is a concatenation of path with a comma.
The second contains the result in our example always 512 and in the last the maximum amount of threads.
```
./LossCalculator -p data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid/output_0.hdf5,data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid/output_1.hdf5 -r 512 -t 8 
```

Again, we provide here also a script to do this automatically for the generated `output_?.hdf5` containers. 

If you have used the shell script from above and generate the voxelgrids with the [generate_tsdf_volumes.py](generate_tsdf_volumes.py).
Then you can now use without changing anything else the script [generate_loss_values.py](generate_loss_values.py).

Be aware, that the four threads already need around 15 GB memory. 
So only increase the amount of the threads if you have enough memory on your system.


