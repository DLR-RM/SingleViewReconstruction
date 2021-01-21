import os
import shutil

import gdown


def download_all_models():
    link_to_file = "https://drive.google.com/uc?id=1PQCpNo1V_RhhMLzZuJUHSPVsBjJv_ZA0"

    current_folder = os.path.abspath(os.path.dirname(__file__))

    output = os.path.join(current_folder, "SingleViewReconstructionModels.zip")
    if not os.path.exists(output):
        gdown.download(link_to_file, output, quiet=False)
    else:
        print("The file is already downloaded!")

    if os.path.exists(output):
        print("Unzipping the file {}".format(os.path.basename(output)))

        os.chdir(current_folder)
        new_model_folder = os.path.join(current_folder, "models")
        if os.path.exists(new_model_folder):
            shutil.rmtree(new_model_folder)
        os.system("unzip {} > /dev/null".format(output))

        print("Move the models in the right places")
        data_folder = os.path.join(current_folder, "data")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        if os.path.exists(new_model_folder):
            # SingleViewReconstruction -> the tree network
            tree_path = os.path.join(new_model_folder, "tree_model")
            goal_path = os.path.join(current_folder, "SingleViewReconstruction", "model")
            if os.path.exists(tree_path) and not os.path.exists(goal_path):
                shutil.move(tree_path, goal_path)

            # autoencoder for the compression and decompression
            ae_path = os.path.join(new_model_folder, "ae_model")
            goal_path = os.path.join(data_folder, "ae_model")
            if os.path.exists(ae_path) and not os.path.exists(goal_path):
                shutil.move(ae_path, goal_path)

            # UNet for the generated normals
            unet_path = os.path.join(new_model_folder, "unet_model")
            goal_path = os.path.join(current_folder, "UNetNormalGen", "model")
            if os.path.exists(unet_path) and not os.path.exists(goal_path):
                shutil.move(unet_path, goal_path)

            shutil.rmtree(new_model_folder)
            os.remove(output)
            print("Downloaded the models for the SingleViewReconstruction, the Autoencoder "
                  "for the comprression and the UNet for the normal generation.")
        else:
            raise Exception("The models folder was not extracted successfully.")

if __name__ == "__main__":
    download_all_models()
