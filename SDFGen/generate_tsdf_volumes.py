
import os
import glob
import subprocess

if __name__ == "__main__":

    # change these paths
    final_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    sdfgen = os.path.abspath(os.path.join(os.path.dirname(__file__), "cmake", "sdfgen"))
    # be aware that the generation per thread needs a lot of memory
    max_threads = 4  # four threads need around 15 GB

    house_paths = glob.glob(os.path.join(final_folder, "*"))

    for house_id_path in house_paths:
        house_obj = os.path.join(house_id_path, "obj", "house.obj")
        house_cam = os.path.join(house_id_path, "cam", "camerapositions")
        if not os.path.exists(house_obj) or not os.path.exists(house_cam):
            print("The house id: {} has a missing obj file "
                  "or missing camera file!".format(os.path.basename(house_id_path)))
        goal_path = os.path.join(house_id_path, "voxelgrid")
        if not os.path.exists(goal_path):
            os.makedirs(goal_path)
        cmd = "{} -o {} -c {} -r 512 -f {} --threads {}".format(sdfgen, house_obj, house_cam, goal_path, max_threads)
        subprocess.call(cmd, shell=True)


