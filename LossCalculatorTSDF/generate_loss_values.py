
import os
import glob
import subprocess

if __name__ == "__main__":

    # change these paths
    final_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    loss_calc = os.path.abspath(os.path.join(os.path.dirname(__file__), "cmake", "LossCalculator"))
    # be aware that the generation per thread needs a lot of memory
    max_threads = 4

    house_paths = glob.glob(os.path.join(final_folder, "*"))

    for house_id_path in house_paths:
        outputs = glob.glob(os.path.join(house_id_path, "voxelgrid", "output_*.hdf5"))
        # filter out results
        outputs = [path for path in outputs if not "loss_avg" in path]
        # filter out already existing loss avg
        outputs = [path for path in outputs if not os.path.exists(path.replace(".hdf5", "_loss_avg.hdf5"))]
        if len(outputs) == 0:
            continue
        combined_file_path = []
        for output in outputs:
            combined_file_path.append(os.path.abspath(output))
        combined_file_path = ",".join(combined_file_path)
        cmd = "{} -r 512 -p {} -t {}".format(loss_calc, combined_file_path, max_threads)
        subprocess.call(cmd, shell=True)


