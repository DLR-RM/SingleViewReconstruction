

from engine.RenderObject import TSDFRenderObject, TSDFRenderObjectTrueAndPrediction
from engine.WindowManager import WindowManager
import h5py
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser("Visualizing of a TSDF grid stored in a .hdf5 container.")
parser.add_argument("path", help="Path to the .hdf5 file.")
parser.add_argument("--voxel_key", help="The key used in the .hdf5 container for the voxelgrid.", default="voxelgrid")
parser.add_argument("--image_key", help="The key used in the .hdf5 container for the color image.", default="colors")
args = parser.parse_args()

file_path = args.path

if os.path.exists(file_path):
    wm = WindowManager((1920, 1024), 'Please wait until the loading is done!')
    wm.init()

    with h5py.File(file_path, "r") as file:
        if args.voxel_key not in file:
            raise Exception("The file must contain a \"{}\"!".format(args.voxel_key))
        voxel = np.array(file[args.voxel_key])
        if voxel.dtype == np.uint16:
            voxel = voxel.astype(np.double) / np.iinfo(np.uint16).max * 0.1 * 2 - 0.1
        image = None
        if args.image_key in file:
            image = np.array(file[args.image_key]).astype(np.uint8)
        else:
            img_path = file_path.replace("voxelgrid", "blenderproc").replace("output_", "")
            if os.path.exists(img_path):
                with h5py.File(img_path, "r") as color_img_file:
                    if "colors" in color_img_file.keys():
                        print("Color image was found automatically!")
                        image = np.array(color_img_file["colors"]).astype(np.uint8)
        if image is None:
            image = ((np.ones((512, 512, 3)) * 0.5) * 255).astype(np.uint8)

    tsdf_render_obj = TSDFRenderObject(voxel, image[:,:,:3], 'first_obj', move_vec=[0, 0, 0])
    wm.add_render_object(tsdf_render_obj)
    voxel = None
    image = None
else:
    raise Exception("File missing: {}".format(file_path))

print("Done loading!")
print("##########################")
help = "Overview over control:\n\tt - changes the texture between image and the normals of the surfaces\n\tu - " \
       "switching between projected and unprojected view\n\to - brings the camera back to the origin\n\tc - " \
       "collapses a prediction and the GT only possible if a TSDFRenderObjectTrueAndPrediction is used.\n\t" \
       "wasd and mouse - for game like movement in the scene."
print(help)
print("##########################")
wm.run_window()
wm = None

