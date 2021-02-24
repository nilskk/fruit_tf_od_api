import time
import os
import random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.csv_util import write_metrics
from utils.file_util import read_tfrecord
from absl import flags
import glob

flags.DEFINE_string('export_path', None, 'Path to exported model')
flags.DEFINE_string('tfrecord_path', './data/tfrecords/vott_val.tfrecord',
                    'Path to tfrecord')

FLAGS = flags.FLAGS


def inference_on_single_image(model, image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]
    tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

    start = time.time()
    detections = model(image_tensor)
    end = time.time()
    time_diff = end - start
    return time_diff


def inference(export_path,
              tfrecord_path):
    tf.config.set_visible_devices([], 'GPU')

    saved_model_path = os.path.join(export_path, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    time_diffs = []
    records = read_tfrecord(tfrecord_path)

    for record in records:
        time_diff = inference_on_single_image(model=model, image=record['image'])
        print(time_diff)
        time_diffs.append(time_diff)

    head, tail = os.path.split(str(export_path))

    time_diffs_array = np.array(time_diffs)

    average_inference_speed = np.average(time_diffs_array[1:])
    print("Average Inference Speed: " + str(average_inference_speed))

    metrics = {'Inference Speed': average_inference_speed}
    write_metrics(head, metrics)


def main(argv):
    flags.mark_flag_as_required('export_path')

    inference(export_path=FLAGS.export_path, tfrecord_path=FLAGS.tfrecord_path)


if __name__ == "__main__":
    tf.compat.v1.app.run(main)
