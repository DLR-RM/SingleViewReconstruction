
import os
import glob
import subprocess

if __name__ == "__main__":

    # change these paths
    final_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    encode_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "encode.py"))

    house_paths = glob.glob(os.path.join(final_folder, "*"))

    for house_id_path in house_paths:
        voxelgrid_path = os.path.join(house_id_path, "voxelgrid")

        cmd = "python {} {}".format(encode_script, voxelgrid_path)
        subprocess.call(cmd, shell=True)


