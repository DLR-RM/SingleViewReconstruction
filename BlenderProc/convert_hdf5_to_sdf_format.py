import json
import os
import h5py
import numpy as np
import argparse


def to_string(l):
    return " ".join([str(e) for e in l])


def convert_hdf5_to_sdf_format(file_path):
    with h5py.File(file_path, "r") as file:
        if "campose" not in file:
            raise Exception("You have to use the CameraStateWriter in your BlenderProc config pipeline.")

        cam_pose = json.loads(np.array(file["campose"]).tostring())[0]
        # is fixed from the SUNCG Camera Sampling
        cam_pose_fov_x = 0.5
        cam_pose_fov_y = 0.388863

        rotation_forward_vec = np.array([float(e) for e in cam_pose["rotation_forward_vec"]])
        rotation_up_vec = np.array([float(e) for e in cam_pose["rotation_up_vec"]])


        new_pose = "  ".join([to_string(cam_pose["location"]), to_string(rotation_forward_vec),
                              to_string(rotation_up_vec), str(cam_pose_fov_x), str(cam_pose_fov_y)])
    return new_pose


def save_convert_hdf5_to_sdf_format(file_path):
    new_pose = convert_hdf5_to_sdf_format(file_path)
    file_name = os.path.basename(file_path)
    new_path = os.path.join(os.path.dirname(file_path), file_name[:file_name.rfind(".")] + "_camera_positions.txt")
    with open(new_path, "w") as file:
        file.write(new_pose + "\n")
    return new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert an .hdf5 saved camera position to the SUNCG camera format, also used in SDFGen.")
    parser.add_argument("file", help="Path to the .hdf5 file.", type=str)
    args = parser.parse_args()
    if args.file is not None and os.path.exists(args.file):
        save_convert_hdf5_to_sdf_format(args.file)
