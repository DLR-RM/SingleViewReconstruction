

import os
import glob
import subprocess
from multiprocessing import Process
from multiprocessing import Semaphore


def generate_obj_and_cam(house_path, index, sema):
    """
    This function generates the house.obj and camera position file for a given house.json file 
    
    :param house_path: A path to a house json file
    :param index: just for printing an nice overview how many houses have been generated so far
    :param sema: a semaphore to process several files in parallel
    """
    global scn2cam, scn2scn, final_folder

    house_id = os.path.basename(os.path.dirname(house_path))
    goal_folder = os.path.join(final_folder, house_id)
    print("Working on: {} is: {}".format(house_id, index))
    os.chdir(os.path.dirname(house_path))

    obj_folder = os.path.join(goal_folder, "obj")
    cam_folder = os.path.join(goal_folder, "cam")
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)

    cmd = " ".join([scn2scn, house_path, os.path.join(obj_folder, "house.obj")]) + " > /dev/null"
    subprocess.call(cmd, shell=True)
    if not os.path.exists(cam_folder):
        os.makedirs(cam_folder)

    cmd = " ".join([scn2cam, house_path, os.path.join(cam_folder, "camerapositions")]) + " > /dev/null"
    subprocess.call(cmd, shell=True)
    sema.release()


if __name__ == "__main__":

    # change these paths
    suncg_toolbox_bin_path = "/home/max/workspace/SUNCGtoolbox/gaps/bin/x86_64"
    suncg_house_folder = "/home/max/Downloads/version_1.1.0/house"
    # this folder will contain a list of house ids in the end
    final_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    # maximum amount of threads, which should be used
    max_threads = 8


    scn2scn = os.path.join(suncg_toolbox_bin_path, "scn2scn")
    scn2cam = os.path.join(suncg_toolbox_bin_path, "scn2cam")
    if not os.path.exists(scn2scn):
        raise Exception("The scn2scn from the SUNCGtoolbox could not be found, make sure you have downloaded "
                        "the SUNCGtoolbox and build the sc2scn file.")

    house_paths = glob.glob(os.path.join(suncg_house_folder, "*", "house.json"))

    print("Generate house.obj and camerapositions for: {} house.json files".format(len(house_paths)))
    sema = Semaphore(max_threads)
    all_processes = []
    for index, house_path in enumerate(house_paths):
        sema.acquire()
        p = Process(target=generate_obj_and_cam, args=(house_path, index, sema))
        all_processes.append(p)
        p.start()

    # inside main process, wait for all processes to finish
    for p in all_processes:
        p.join()


