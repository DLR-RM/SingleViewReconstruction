import os
import subprocess

if __name__ == "__main__":
    # This script can only be executed after some house.obj AND camerapositions have been generated
    # Check the SDFGen folder more information on that.

    # you need to download BlenderProc from:
    # https://github.com/DLR-RM/BlenderProc, make sure that you use 1.7.0
    blenderproc_folder = "/home/max/workspace/BlenderProc"
    suncg_house_folder = "/home/max/Downloads/version_1.1.0/house"
    final_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    # config file for BlenderProc
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))

    if not os.path.exists(final_folder):
        raise Exception("The data folder does not exist yet. Check the SDFGen folder for more information.")


    for house_id in os.listdir(final_folder):
        camposes = os.path.join(final_folder, house_id, "cam", "camerapositions")
        if not os.path.exists(camposes):
            print("The house id: {} does not have a camera positions file".format(house_id))
            continue

        suncg_house_json_file = os.path.join(suncg_house_folder, house_id, "house.json")
        if not os.path.exists(suncg_house_json_file):
            print("The house json file could not be found, make sure the suncg path is correct!")
            continue

        goal_folder = os.path.join(final_folder, house_id, "blenderproc")

        cmd = "python {} {} {} {} {}".format(os.path.join(blenderproc_folder, "run.py"), config_file,
                                             camposes, suncg_house_json_file, goal_folder)
        subprocess.call(cmd, shell=True)








