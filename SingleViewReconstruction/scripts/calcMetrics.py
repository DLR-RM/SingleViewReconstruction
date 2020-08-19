import argparse
import os
import time

import numpy as np
import h5py
import uuid

from skimage import measure
import subprocess


def convert_to_float(data, threshold=0.1):
    """
    Converts the uint16 stored voxelgrids to float again
    :param data: the voxelgrid
    :param threshold: the new used threshold
    :return: a converted voxelgrid
    """
    if data.dtype == np.uint16:
        return data.astype(np.float) / float(np.iinfo(np.uint16).max) * threshold * 2 - threshold
    return data


def calc_iou(true, prediction):
    """
    Calculates the IOU between a true voxelgrid and predicted voxelgrid
    :param true: GT voxelgrid
    :param prediction: predicted voxelgrid
    :return: iou, abs_diff, occupied, precision, recall
    """
    tsdf_val = 0
    true_occ = true <= tsdf_val
    test_occ = prediction <= tsdf_val
    first = np.count_nonzero(np.logical_and(true_occ,  test_occ))
    true_positive = first
    sec = np.count_nonzero(np.logical_or(true_occ, test_occ))
    all_labeled_positive = np.count_nonzero(test_occ)
    all_true_positive = np.count_nonzero(true_occ)
    if all_labeled_positive > 0:
        precision = true_positive / float(all_labeled_positive)
    else:
        precision = -1.0
    if all_true_positive > 0:
        recall = true_positive / float(all_true_positive)
    else:
        recall = -1.0
    c_size = 512.0**3
    occupied = np.count_nonzero(true_occ) / float(c_size)
    iou = float(first) / float(sec)
    abs_diff = np.mean(np.abs(true-prediction))
    return iou, abs_diff, occupied, precision, recall


def generate_vertices_obj(data, output_file_path, threshold_val=0.0):
    """
    This function generates based on a voxel grid and an given output path an obj file, which contains all vertices
    on the surface of the TSDF.
    :param data: TSDF field
    :param output_file_path: path where the resulting .obj file should be stored
    :param threshold_val: Where the surface should be constructed usually 0.0
    :return:
    """
    if np.max(data) < threshold_val or np.min(data) > threshold_val:
        print("Max and min are: " + str(np.max(data)) + " " + str(np.min(data)))
        return
    # data = np.flip(data, axis=2)
    verts, _, _, _ = measure.marching_cubes(data, threshold_val)
    verts *= 1. / np.max(data.shape)
    text = "\n".join(["v {} {} {}".format(v[0], v[1], v[2]) for v in verts])
    with open(output_file_path, 'w') as file:
        file.write(text)
    return verts.shape[0]


def calc_hausdorff_distance(true_file, test_file, amount_of_true_vertices, meshlab_path, unique_id=None):
    """
    Calculates the house dorf distance for a two .obj files, which contain the vertices of the surface
    :param true_file: path to an .obj file
    :param test_file: path to an .obj file
    :param amount_of_true_vertices: amount of vertices in the true file
    :param meshlab_path: Path to the meshlabserver executable
    :param unique_id: unique id to avoid clashes
    :return:
    """
    part = '<!DOCTYPE FilterScript>\n<FilterScript>\n <filter name="Hausdorff Distance">\n  ' \
           '<Param tooltip="The mesh whose surface is sampled. For each sample we search the closest point on the ' \
           'Target Mesh." type="RichMesh" name="SampledMesh" isxmlparam="0" description="Sampled Mesh" value="0"/>\n' \
           '  <Param tooltip="The mesh that is sampled for the comparison." type="RichMesh" name="TargetMesh" ' \
           'isxmlparam="0" description="Target Mesh" value="1"/>' + \
           '  <Param tooltip="Save the position and distance of all the used samples on both the two surfaces, ' \
           'creating two new layers with two point clouds representing the used samples." type="RichBool" ' \
           'name="SaveSample" isxmlparam="0" description="Save Samples" value="false"/>\n' + \
           '  <Param tooltip="For the search of maxima it is useful to sample vertices and edges of the mesh with ' \
           'a greater care. It is quite probably the the farthest points falls along edges or on mesh vertices, and ' \
           'with uniform montecarlo sampling approachesthe probability of taking a sample over a vertex or an edge ' \
           'is theoretically null.&lt;br>On the other hand this kind of sampling could make the overall sampling ' \
           'distribution slightly biased and slightly affects the cumulative results." type="RichBool" ' \
           'name="SampleVert" isxmlparam="0" description="Sample Vertices" value="true"/>\n' + \
           '  <Param tooltip="See the above comment." type="RichBool" name="SampleEdge" isxmlparam="0" ' \
           'description="Sample Edges" value="false"/>\n' + \
           '  <Param tooltip="See the above comment." type="RichBool" name="SampleFauxEdge" isxmlparam="0" ' \
           'description="Sample FauxEdge" value="false"/>\n' + \
           '  <Param tooltip="See the above comment." type="RichBool" name="SampleFace" isxmlparam="0" ' \
           'description="Sample Faces" value="false"/>\n'
    part += '  <Param tooltip="The desired number of samples. It can be smaller or larger than the mesh size, ' \
            'and according to the choosed sampling strategy it will try to adapt." type="RichInt" name="SampleNum" ' \
            'isxmlparam="0" description="Number of samples" value="{}"/>\n'.format(amount_of_true_vertices)
    part += '  <Param tooltip="Sample points for which we do not find anything within this distance are rejected ' \
            'and not considered neither for averaging nor for max." max="1.71684" min="0" type="RichAbsPerc" ' \
            'name="MaxDist" isxmlparam="0" description="Max Distance" value="1.0"/>\n </filter>\n</FilterScript>'
    if unique_id is None:
        unique_id = uuid.uuid4().hex
    tmp_path = '/tmp/meshlab_script_{}.mlx'.format(unique_id)
    log_path = '/tmp/log_{}.txt'.format(unique_id)
    with open(tmp_path, 'w') as file:
        file.write(part)
    cmd = '{} -l {} -i {} -i {} -s {} 1> /dev/null 2> /dev/null '.format(meshlab_path, log_path,
                                                                         true_file, test_file, tmp_path)
    subprocess.call(cmd, shell=True)
    val_min, val_max, mean, RMS = None, None, None, None
    with open(log_path, 'r') as file:
        text = file.read()
        pos = text.find('BBox Diag')
        text = text[:pos]  # delete everything behind the bounding box comparison
        lines = text.split('\n')
        lines = [line for line in lines if 'min :' in line]
        if len(lines) == 1:
            values = lines[0].replace(':', '')
            values = [value for value in values.split(' ') if '.' in value]
            if len(values) == 4:
                #  order is min, max, mean, RMS
                val_min, val_max, mean, RMS = [float(value) for value in values]
            else:
                print("Something went wrong", values)
    os.remove(log_path)
    os.remove(tmp_path)
    return val_min, val_max, mean, RMS


def calc_hausdorff_distance_data(true_data, test_data, meshlab_path):
    """
    Calculates the housedorff distance for two voxelgrids and the meshlabserver path
    :param true_data: GT voxelgrid data
    :param test_data: predicted voxelgrid data
    :param meshlab_path: path to the meshlabserver, usually "/usr/bin/meshlabserver"
    :return: min_hd, max_hd, mean_hd, RMS_hd
    """
    sw = time.time()
    unique_id = uuid.uuid4().hex
    true_vert_path = '/tmp/true_vert_{}.obj'.format(unique_id)
    test_vert_path = '/tmp/test_vert_{}.obj'.format(unique_id)
    generate_vertices_obj(test_data, test_vert_path)
    amount_of_true_vertices = generate_vertices_obj(true_data, true_vert_path)
    res = calc_hausdorff_distance(true_vert_path, test_vert_path, amount_of_true_vertices, meshlab_path, unique_id)
    if os.path.exists(true_vert_path):
        os.remove(true_vert_path)
    if os.path.exists(test_vert_path):
        os.remove(test_vert_path)
    print("Took {} for hausdorff dist".format(time.time() - sw))
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Calculates the hausdorff distance between two .hdf5 containers.")
    parser.add_argument("--true", help="Path to the true .hdf5 file, must contain a voxelgrid.", required=True)
    parser.add_argument("--test", help="Path to the test .hdf5 file, must contain a voxelgrid.", required=True)
    parser.add_argument("--meshlab_path", help="Path to the meshlab server executable.", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.true):
        raise Exception("The true .hdf5 file can not be found: {}".format(args.true))

    if not os.path.exists(args.test):
        raise Exception("The test .hdf5 file can not be found: {}".format(args.test))

    if not os.path.exists(args.meshlab_path):
        raise Exception("The meshlab exectuable can not be found: {}".format(args.meshlab_path))

    with h5py.File(args.true, "r") as file:
        if "voxelgrid" in file.keys():
            true_data = convert_to_float(np.array(file["voxelgrid"]))
        else:
            raise Exception("The true .hdf5 container does not have a voxelgrid!")

    with h5py.File(args.test, "r") as file:
        if "voxelgrid" in file.keys():
            test_data = convert_to_float(np.array(file["voxelgrid"]))
        else:
            raise Exception("The test .hdf5 container does not have a voxelgrid!")

    iou, abs_diff, occupied, precision, recall = calc_iou(true_data, test_data)
    print("IOU and reading done, next is HD calculation")
    val_min_hd, val_max_hd, mean_hd, RMS_hd = calc_hausdorff_distance_data(true_data, test_data, args.meshlab_path)
    print("It has an of iou: {}, precision: {}, recall: {}, diff: {}, occupied: {}, max_hd: {}, "
          "mean_hd: {}, rms_hd: {}".format(iou, precision, recall, abs_diff, occupied, val_max_hd, mean_hd, RMS_hd))
