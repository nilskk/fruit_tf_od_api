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

flags.DEFINE_string('export_dir', None, 'Path to exported model')
flags.DEFINE_string('tfrecord', './data/tfrecords/vott_val.tfrecord',
                    'Path to tfrecord')
flags.DEFINE_string('label_map_path', './data/voc_data/pascal_label_map.pbtxt',
                    'Path to label map proto')

FLAGS = flags.FLAGS

def load_image_as_nparray(path):
    return np.array(Image.open(path))

def inference_on_single_image(model, image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]
    tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

    start = time.time()
    detections = model(image_tensor)
    end = time.time()
    time_diff = end - start
    return time_diff

def inference(argv):
    flags.mark_flag_as_required('export_dir')
    saved_model_path = os.path.join(FLAGS.export_dir, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    time_diffs = []
    records = read_tfrecord(FLAGS.tfrecord)

    for record in records:
        time_diff = inference_on_single_image(model=model, image=record['image'])
        print(time_diff)
        time_diffs.append(time_diff)

    head, tail = os.path.split(str(FLAGS.export_dir))

    time_diffs_array = np.array(time_diffs)

    average_inference_speed = np.average(time_diffs_array[1:])
    print("Average Inference Speed: " + str(average_inference_speed))

    metrics = {'Inference Speed': average_inference_speed}
    write_metrics(head, metrics)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.app.run(inference)
