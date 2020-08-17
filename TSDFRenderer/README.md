
# TSDFRenderer

This module can visualize tsdf volumes stored in `.hdf5` containers.

```shell script
python visualize_tsdf.py ../data/510bb6a0e4dbe783109adc01b05d8c32/voxelgrid/output_0.hdf5
```

This will load the `"voxelgrid"` key from the `.hdf5` container and also search for the color image.
The color image can also be stored in the same file with the key `"colors"` or if the file is in the `data` folder, the corresponding image file is found automatically.

Please wait with pressing a key until the scene is loaded.