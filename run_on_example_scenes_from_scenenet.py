import os
import subprocess
import shutil
import gdown
import glob
import random
import argparse

import h5py

from download_models import download_all_models
from BlenderProc.convert_hdf5_to_sdf_format import convert_hdf5_to_sdf_format


def print_info():
    # print some information, before executing
    info = "This script will perform a lot of steps automatically, including the download of SceneNet, BlenderProc, " \
           "weights of the models used in this repo, and Blender.\n"
    info += "If you agree with the download of the dataset and the open source code BlenderProc, type in yes:"
    agreed = input(info).strip().lower()
    if not (agreed == "yes" or agreed == "y"):
        raise Exception("This can only be executed if agreed to the statement above.")


def download_BlenderProc(blender_proc_path):
    """
    This fct. autonomously downloads BlenderProc. From here https://github.com/DLR-RM/BlenderProc.git
    This is used to generate the color and normal images used for testing.
    :param blender_proc_path: path where BlenderProc will be saved to
    :return blender_proc_git_path: path to the downloaded BlenderProc git project
    """
    blender_proc_git_path = os.path.join(blender_proc_path, "BlenderProc")
    # download BlenderProc
    if not os.path.exists(blender_proc_git_path):
        print("First step is downloading BlenderProc:")
        cmd = "git clone -b 'v1.8.1' --single-branch --depth 1 https://github.com/DLR-RM/BlenderProc.git " \
              "> /dev/null 2> /dev/null"
        subprocess.call([cmd], shell=True, cwd=blender_proc_path)
        print("Download of BlenderProc complete.")
    return blender_proc_git_path


def download_SceneNet(blender_proc_git_path):
    """
    This fct. autonomously downloads SceneNet via the scripts given in BlenderProc. It also downloads the
    texture_folder and unpacks it. This will most likely only work under linux.
    :param blender_proc_git_path path to the downloaded BlenderProc project
    :return scenenet_data_folder: path to the downloaded SceneNet
    """
    # Download the scenenet dataset
    scenenet_data_folder = os.path.join(blender_proc_git_path, "resources", "scenenet", "SceneNetData")
    if not os.path.exists(scenenet_data_folder):
        print("Next the scenenet dataset has to be downloaded, this might take a while.")
        cmd = "python scripts/download_scenenet.py > /dev/null"
        subprocess.call([cmd], shell=True, cwd=blender_proc_git_path)
        print("Done downloading scenenet.")
    # download the scenenet texture folder
    scenenet_texture_folder = os.path.join(blender_proc_git_path, "resources", "scenenet", "texture_library")
    if not os.path.exists(scenenet_texture_folder):
        print("Download the scenenet textures")
        url_for_texture_zip = "https://drive.google.com/uc?id=0B_CLZMBI0zcuQ3ZMVnp1RUkyOFk"
        print("###########################")
        print("You have to download the scenenet texture yourself: {}".format(url_for_texture_zip))
        print("The reason being that the owner changed the access permissions, we are sorry about that.")
        print("###########################")
        output = input("Path to the downloaded texture_library.tgz: ")
        cmd = "tar zxvf {} > /dev/null".format(output)
        subprocess.call([cmd], shell=True, cwd=os.path.join(scenenet_data_folder, ".."))
        os.remove(output)
        # Create the texture folder, for all materials, which are undefined in the scenenet dataset
        print("Sample textures from the other for the unknown texture folder.")
        unknown_folder = os.path.join(scenenet_texture_folder, "unknown")
        if not os.path.exists(unknown_folder):
            os.makedirs(unknown_folder)
            textures = glob.glob(os.path.join(scenenet_texture_folder, "*", "*.jpg"))
            textures.sort()
            used_textures = random.choices(textures, k=5)
            for texture in used_textures:
                shutil.copyfile(texture, os.path.join(unknown_folder, os.path.basename(texture)))
        print("Done with the downloading of the scenenet texture.")
    return scenenet_data_folder


def render_some_images_with_blenderproc(blender_proc_git_path, scenenet_data_folder, output_dir, used_seed):
    """
    Render some images with BlenderProc of the downloaded SceneNet dataset.

    :param blender_proc_git_path: path to the downloaded BlenderProc project
    :param scenenet_data_folder: path to the downloaded SceneNet dataset
    :param output_dir: path where the resulting .hdf5 files containing the color and normal images should be saved
    :param used_seed: a random seed to make the results repeatable
    :return image_output_folder, used_object_file: the path to the folder where the .hdf5 files will be saved, \
                                                   the path to the used .obj file used in this example
    """
    used_object_file = os.path.join(scenenet_data_folder, "1Bedroom", "bedroom_1.obj")
    image_output_folder = os.path.join(output_dir, "images")
    if not os.path.exists(output_dir):
        print("Render a few images with BlenderProc")
        used_env = dict(os.environ)
        used_env["BLENDER_PROC_RANDOM_SEED"] = str(used_seed)
        cmd = "python run.py ../scenenet_config.yaml {} resources/scenenet/texture_library " \
              "{} > /dev/null".format(used_object_file, image_output_folder)
        subprocess.call([cmd], shell=True, cwd=blender_proc_git_path, env=used_env)
        print("Done with the rendering")
    return image_output_folder, used_object_file

def generate_ground_truth(image_output_folder, used_object_file):
    """
    Generate the ground truth, for that it is necessary that the CMakeLists.txt was adapted correctly.
    Check the SDFGEN/README.md for more information.
    :param image_output_folder the path to the folder where the .hdf5 files will be saved
    :param used_object_file: the path to the used .obj file used in this example
    """
    camera_pose_file = os.path.join(image_output_folder, "camera_positions")
    # generate the corresponding TSDF volumes by first generating the camera poses in the correct format
    if os.path.exists(image_output_folder):
        imgs_paths = glob.glob(os.path.join(image_output_folder, "*.hdf5"))
        new_poses = [""] * len(imgs_paths)
        for img_path in imgs_paths:
            use_this_file = False
            with h5py.File(img_path, "r") as file:
                if "campose" in file:
                    use_this_file = True
            if use_this_file:
                new_pose = convert_hdf5_to_sdf_format(img_path)
                number = int(os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")])
                new_poses[number] = new_pose
        with open(camera_pose_file, "w") as file:
            file.write("\n".join(new_poses))

    # generate the TSDF volumes by using the camera pose and the object
    if os.path.exists(camera_pose_file) and os.path.exists(used_object_file):
        sdfgen = os.path.join(main_folder, "SDFGen", "cmake", "sdfgen")
        if not os.path.exists(sdfgen):
            print("This will now build the SDFGen project. If you haven't change the CMakeLists.txt, this will fail!")
            agreed = input("I have changed the SDFGen/CMakeLists.txt, then type in yes:").strip().lower()
            if not (agreed == "yes" or agreed == "y"):
                raise Exception("You need to change the SDFGen/CMakeLists.txt before generating the ground truth.")
            # build the project
            cmd = "mkdir cmake && cd cmake && cmake -DCMAKE_BUILD_TYPE=RELEASE ..  && make -j 8"
            subprocess.call([cmd], shell=True, cwd=os.path.join(main_folder, "SDFGen"))
            if not os.path.exists(sdfgen):
                raise Exception("The building of the sdfgen failed")
        print("Generate the TSDF volumes, this might take a while.")
        if not os.path.exists(true_voxel_dir):
            os.makedirs(true_voxel_dir)
        cmd = "{} -o {} -c {} -r 512 -f {} --threads {} > /dev/null".format(sdfgen, used_object_file, camera_pose_file,
                                                                            true_voxel_dir, max_threads)
        subprocess.call([cmd], shell=True, cwd=os.path.join(main_folder, "SDFGen"))


def download_models(main_folder):
    """
    This downloads all models used in this project.
    :param main_folder: The folder in which this file is located
    :return unet_folder: returns the path to the UNetNormalGen dir
    """
    # check if the models are already there, if not download them
    unet_folder = os.path.abspath(os.path.join(main_folder, "UNetNormalGen"))
    model_folder = os.path.join(unet_folder, "model")
    if not os.path.exists(model_folder):
        print("Download all models, by running the 'download_models.py'")
        download_all_models()
    return unet_folder

def generate_normals_with_unet(unet_folder, generate_normals):
    """
    Generate the normals with the UnetNormalGen network.
    :param generate_normals: if this is true the normals are generated
    """
    # if this is done with generated normals, the UNet is used to generate them
    if generate_normals:
        cmd = "python generate_predicted_normals.py --model_path model/model.ckpt " \
              "--path ../BlenderProc/output_dir/images/@.hdf5 --use_pretrained 2> /dev/null"
        subprocess.call([cmd], shell=True, cwd=unet_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Whole pipeline run on the SceneNet. This will download BlenderProc/Blender and "
                                     "SceneNet and all models used in this project.")
    parser.add_argument("--use_generated_normals", help="If this is used, the UNetModel is used to generate normals, "
                                                        "which are used instead of the synthetic ones.",
                        action="store_true")
    parser.add_argument("--generate_ground_truth", help="If this is used, the ground truth TSDF volumes are "
                                                        "calculated, make sure that you changed the "
                                                        "SDFGen/CMakeLists.txt before using this!",
                        action='store_true')
    parser.add_argument("--amount_of_threads", help="If ground truth data is generated, it needs a lot of RAM, each "
                                                    "thread needs around 4GB.", type=int, default=4)
    args = parser.parse_args()
    print_info()

    # four threads need around 15 GB, scale this value down if you have less than 15 GB of RAM
    max_threads = args.amount_of_threads

    main_folder = os.path.abspath(os.path.dirname(__file__))
    blender_proc_path = os.path.abspath(os.path.join(main_folder, "BlenderProc"))
    output_dir = os.path.join(blender_proc_path, "output_dir")
    true_voxel_dir = os.path.join(output_dir, "true_voxels")

    # if this is False the TreeNetwork will use the synthetic generated Normal Img.
    generate_normals = args.use_generated_normals

    # set this only to true if you are sure that the CMakeLists.txt in SDFGen is adjusted to your PC
    should_generate_ground_truth = args.generate_ground_truth

    used_seed = 1
    random.seed(used_seed)

    # delete all pre-existing files, from previous runs of this script
    if os.path.exists(output_dir):
        print("Remove all outputs previously generated")
        shutil.rmtree(output_dir)

    # download BlenderProc
    blender_proc_git_path = download_BlenderProc(blender_proc_path)

    # download the full dataset SceneNet with corresponding textures
    scenenet_data_folder = download_SceneNet(blender_proc_git_path)

    # Render four images in the SceneNet dataset
    image_output_folder, used_object_file = render_some_images_with_blenderproc(blender_proc_git_path,
                                                                                scenenet_data_folder, output_dir,
                                                                                used_seed)
    if should_generate_ground_truth:
        generate_ground_truth(image_output_folder, used_object_file)

    unet_folder = download_models(main_folder)
    generate_normals_with_unet(unet_folder, generate_normals)

    # Finally predict for each of the four images a 3D scene and save it
    cmd = "python predict_datapoint.py ../BlenderProc/output_dir/images/@.hdf5 --output ../BlenderProc/output_dir/ " \
          "--use_pretrained_weights 2> /dev/null"
    if generate_normals:
        cmd += " --use_gen_normal"
    subprocess.call([cmd], shell=True, cwd=os.path.join(main_folder, "SingleViewReconstruction"))

    print("Done with creating the example outputs, you can view them with:")

    print("python TSDFRenderer/visualize_tsdf.py BlenderProc/output_dir/output_0.hdf5\n")
    if should_generate_ground_truth:
        print("The corresponding true label can be viewed with this: ")
        print("python TSDFRenderer/visualize_tsdf.py BlenderProc/output_dir/true_voxels/output_0.hdf5")
