
import os
import h5py
import glob
import tensorflow as tf
import threading
from src.utils.stopwatch import StopWatch
from src.utils.averagenumber import AverageNumber
from src.utils.timeframe import TimeFrame
import numpy as np
import sys
import time

max_threads = 8
goal_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

if not os.path.exists(data_folder):
    raise Exception("The data folder is missing!")

color_normal_mean_path = os.path.join(goal_folder, "color_normal_mean.hdf5")

with h5py.File(color_normal_mean_path, 'r') as data:
    normal_mean_img = np.array(data['normal'])

def serialize_tfrecord(color, normal, encoded, lossmap):
    feature = {}
    feature['color'] = tf.train.Feature(float_list=tf.train.FloatList(value=color.reshape(-1)))
    feature['normal'] = tf.train.Feature(float_list=tf.train.FloatList(value=normal.reshape(-1)))
    feature['voxel'] = tf.train.Feature(float_list=tf.train.FloatList(value=encoded.reshape(-1)))
    feature['lossmap'] = tf.train.Feature(float_list=tf.train.FloatList(value=lossmap.reshape(-1)))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    return serialized

list_of_done_elements = []
lock_writer_nr = threading.Lock()
lock_done = threading.Lock()
done = False
sem_done = threading.Semaphore(0)
done_counter = 0
total_length = 10
global_write_counter = 0

mean_for_serialize = AverageNumber()
mean_for_write = AverageNumber()
mean_for_read = AverageNumber()

def write_fct():
    global sem_done, global_write_counter, done

    amount_per_tf_record = 100
    options = tf.io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = None
    while not done:
        sem_done.acquire()
        try:
            lock_done.acquire()
            if len(list_of_done_elements) > 0:
                c, n, o, l = list_of_done_elements.pop()
            else:
                lock_done.release()
                continue
            lock_done.release()
        finally:
            pass
        if global_write_counter % amount_per_tf_record == 0:
            if writer is not None:
                writer.close()
            train_writer_nr = global_write_counter // amount_per_tf_record
            writer = tf.io.TFRecordWriter(os.path.join(goal_folder, "train_{}.tfrecord".format(train_writer_nr)), options=options)
            print("Start train_{}.tfrecord file".format(train_writer_nr))
        sw = StopWatch()
        serialized = serialize_tfrecord(c, n, o, l)
        mean_for_serialize.add_new(sw.elapsed_time_val)
        sw = StopWatch()
        writer.write(serialized)
        mean_for_write.add_new(sw.elapsed_time_val)
        global_write_counter += 1
    print("Writer closed")
    writer.close()

def read_fct():
    global done_counter, list_of_done_elements, lock_done, normal_mean_img, total_length
    search_path = os.path.join(data_folder, "*", "voxelgrid", "output_*.hdf5")
    paths = glob.glob(search_path)
    paths = [path for path in paths if "_loss_avg.hdf5" not in path]
    total_length = len(paths)
    true_total_length = 0
    for encoded_voxel_path in paths:
        output_o, loss_o, color_o, normal_o = None, None, None, None
        try:
            loss_avg_path = encoded_voxel_path.replace('.hdf5', '_loss_avg.hdf5')
            blender_result_path = encoded_voxel_path.replace('voxelgrid', 'blenderproc').replace('output_', '')
            sw = StopWatch()
            if os.path.exists(encoded_voxel_path) and os.path.exists(loss_avg_path) and os.path.exists(blender_result_path):
                with h5py.File(encoded_voxel_path, 'r') as data:
                    if 'encoded_voxelgrid' in data.keys():
                        output_o = np.array(data["encoded_voxelgrid"]).astype(np.float32)
                    else:
                        continue
                with h5py.File(loss_avg_path, 'r') as data:
                    if 'lossmap' in data.keys():
                        loss_o = np.array(data["lossmap"]).astype(np.float32)
                with h5py.File(blender_result_path, 'r') as data:
                    if "colors" in data.keys() and 'normals' in data.keys():
                        raw_image = np.array(data['colors'])
                        amount_of_elements = raw_image.shape[0] * raw_image.shape[1] * raw_image.shape[2]
                        if np.count_nonzero(raw_image) < amount_of_elements * 0.80:
                            continue
                        else:
                            color_o = raw_image
                        color_o = color_o[:, :, :3].astype(np.uint8)
                        normal_o = np.array(data["normals"])
                    else:
                       continue
            needed = sw.elapsed_time_val
            mean_for_read.add_new(needed)
        except IOError:
            continue
        if output_o is not None and loss_o is not None and color_o is not None and normal_o is not None:
            normal_o -= normal_mean_img

            # make channel first
            normal_o = np.transpose(normal_o, [2,0,1])
            output_o = np.transpose(output_o, [3,0,1,2])
            lock_done.acquire()
            list_of_done_elements.append([color_o, normal_o, output_o, loss_o])
            done_counter += 1
            lock_done.release()
            sem_done.release()
            true_total_length += 1
    total_length = true_total_length
    print("Reader done")


threads = []

t = threading.Thread(target=write_fct)
t.daemon = True
threads.append(t)
t.start()

t = threading.Thread(target=read_fct)
t.daemon = True
threads.append(t)
t.start()

update_nr = 0
while not done:
    lock_done.acquire()
    lenght_now = len(list_of_done_elements)
    lock_done.release()
    max_time = np.max([mean_for_read.mean, mean_for_write.mean + mean_for_serialize.mean]) / 6.0
    print("\r"+str(update_nr)+" At: " + str(global_write_counter + 1) + " of " + str(total_length) + ", done: " + str(done_counter) + ", avg serialize: " + str(TimeFrame(mean_for_serialize.mean)) + ", rest: " + str(TimeFrame(float(total_length - global_write_counter) * max_time)) + ', avg read: ' + str(TimeFrame(mean_for_read.mean)) + ', avg write: ' + str(TimeFrame(mean_for_write.mean)) + ', len: ' + str(lenght_now) + '                      ', end='')
    update_nr += 1
    sys.stdout.flush()
    time.sleep(1)
    if global_write_counter >= total_length:
        done = True
        break
print("Done with updating waiting for finish of loops!")
for thread in threads:
    thread.join()

print("All done")
