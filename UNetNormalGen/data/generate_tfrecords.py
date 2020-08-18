from __future__ import print_function

import os
import numpy as np
import h5py
import cv2
import tensorflow as tf
import sys
import threading
import time
import glob
import argparse

parser = argparse.ArgumentParser(description="Creates .tfrecord")
parser.add_argument('--path', type=str, help="The path where to look for .h5py files. Use @ for * in the path.", required=True)
parser.add_argument('--out', type=str, help="The path where to write the tfrecords.", required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ''

folder_path = args.path
write_folder_path = args.out

image_size = (512, 512)
def resize_image(img):
    return cv2.resize(img, image_size)

def serialize_tfrecord(color, normal):
    feature = {}
    feature['colors'] = tf.train.Feature(float_list=tf.train.FloatList(value=color.reshape(-1)))
    feature['normals'] = tf.train.Feature(float_list=tf.train.FloatList(value=normal.reshape(-1)))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    return serialized

def deserialize_tfrecord(example_proto):
    keys_to_features = {'colors': tf.FixedLenFeature([], tf.string),
                        'normals': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    color = tf.cast(tf.reshape(tf.decode_raw(parsed_features['colors'], tf.uint8), (512, 512, 3)), tf.float32)
    normal = tf.reshape(parsed_features['normals'], (512, 512, 3))

    return (color, normal)


list_of_open_elements = []
list_of_done_elements = []
lock_open = threading.Lock()
lock_done = threading.Lock()
lock_writer = threading.Lock()
lock_writer_nr = threading.Lock()
sem_open = threading.Semaphore(0)
sem_done = threading.Semaphore(0)
done = False
done_counter = 0
global_write_counter = 0


def normalizing_fct():
    global list_of_open_elements, sem_open, done_counter
    counter_here = 0
    while not done:
        shouldSleep = True
        while shouldSleep and not done:
            try:
                lock_done.acquire()
                size = len(list_of_done_elements)
                lock_done.release()
            finally:
                pass
            shouldSleep = size > 100
            if shouldSleep:
                time.sleep(0.25)
        if done:
            break
        sem_open.acquire()
        if done:
            break
        try:
            lock_open.acquire()
            path = list_of_open_elements.pop()
            lock_open.release()
        finally:
            pass
        time.sleep(0.05)
        try:
            with h5py.File(path, 'r') as data:
                if "colors" in data.keys() and "normals" in data.keys():
                    raw_image = np.array(data["colors"])
                    raw_image = raw_image[:, :, :3]
                    color_o = (raw_image / 255.0)
                    normal_o = np.array(data["normals"])
                    # converting normal images
                    if np.any(np.isinf(normal_o)) or np.min(normal_o) < 0:
                        print("This .hdf5 container contains an invalid normal img: {}".format(path))
                        continue
                    counter_here += 1
                else:
                    continue
        except IOError:
            continue

        lock_done.acquire()
        list_of_done_elements.append([color_o, normal_o])
        done_counter += 1
        lock_done.release()
        color_o = None
        normal_o = None
        sem_done.release()

writer_nr = 0


def write_fct():
    global sem_done, global_write_counter, writer_nr
    start_new_writer = True
    writer = None
    write_counter = 0
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    while not done:
        if start_new_writer:
            lock_writer_nr.acquire()
            if writer is not None:
                writer.close()
            writer = None
            writer = tf.python_io.TFRecordWriter(
                os.path.join(write_folder_path, "train_{}.tfrecord".format(writer_nr)), options=options)
            print("Start new tf record file")
            writer_nr += 1
            start_new_writer = False
            lock_writer_nr.release()
        sem_done.acquire()
        try:
            lock_done.acquire()
            if len(list_of_done_elements) > 0:
                c, n = list_of_done_elements.pop()
            else:
                lock_done.release()
                continue
            lock_done.release()
        finally:
            pass
        serialized = serialize_tfrecord(c, n)
        lock_writer.acquire()
        writer.write(serialized)
        global_write_counter += 1
        lock_writer.release()
        serialized = None
        c = None
        n = None
        write_counter += 1
        if write_counter % 500 == 0:
            start_new_writer = True

    print("Done writing fct: " + str(write_counter))
    writer.close()


paths = glob.glob(os.path.join(folder_path.replace("@", "*"), "*.hdf5"))
paths.sort()
paths.reverse()

threads = []
normalizer_workers = 1
writer_workers = 1

for i in range(writer_workers):
    t = threading.Thread(target=write_fct)
    t.daemon = True
    threads.append(t)
    t.start()

for i in range(normalizer_workers):
    # Create each thread, passing it its chunk of numbers to factor
    # and output dict.
    t = threading.Thread(target=normalizing_fct)
    t.daemon = True
    threads.append(t)
    t.start()


counter = len(paths)
for path in paths:
    lenght_ok = True
    while lenght_ok:
        lock_done.acquire()
        lenght_now = len(list_of_done_elements)
        lock_done.release()
        lenght_ok = False
        if lenght_now > 15:
            time.sleep(0.1)
            lenght_ok = True
    lock_open.acquire()
    list_of_open_elements.append(path)
    lock_open.release()
    sem_open.release()
while not done:
    lock_done.acquire()
    lenght_now = len(list_of_done_elements)
    lock_done.release()
    lock_open.acquire()
    length_2 = len(list_of_open_elements)
    lock_open.release()
    sys.stdout.flush()
    time.sleep(0.25)
    if lenght_now + length_2 == 0:
        done = True
        break
print(' ')
for i in range(writer_workers):
    sem_done.release()
for i in range(writer_workers):
    threads[i].join()
print("Writer closed")
sem_open.release()
for t in threads:
    t.join()
exit()
