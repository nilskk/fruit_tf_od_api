import time
import os
import random
import tensorflow as tf
from PIL import Image
import numpy as np
from csv_util import write_speed
from absl import flags
import glob

flags.DEFINE_string('export_dir', None, 'Path to exported model')
flags.DEFINE_string('test_image_dir', './data/test_data/images',
                    'Path to test images dir')
flags.DEFINE_string('label_map_path', './data/voc_data/pascal_label_map.pbtxt',
                    'Path to label map proto')

FLAGS = flags.FLAGS

def load_image_as_nparray(path):
    return np.array(Image.open(path))

def inference(argv):
    flags.mark_flag_as_required('export_dir')
    saved_model_path = os.path.join(FLAGS.export_dir, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    num_images = len(glob.glob(FLAGS.test_image_dir + '/**/*', recursive=True))
    print(num_images)
    time_diffs = []

    for i, image_name in enumerate(random.sample(os.listdir(FLAGS.test_image_dir), k=num_images)):
        image_path = os.path.join(FLAGS.test_image_dir, image_name)
        image_np = load_image_as_nparray(image_path)
        image_tensor = tf.convert_to_tensor(image_np)
        image_tensor = image_tensor[tf.newaxis, ...]
        tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

        start = time.time()
        detections = model(image_tensor)
        end = time.time()
        time_diff = end - start
        print(time_diff)
        time_diffs.append(time_diff)

    head, tail = os.path.split(str(FLAGS.export_dir))

    time_diffs_array = np.array(time_diffs)

    average_inference_speed = np.average(time_diffs_array[1:])
    print("Average Inference Speed: " + str(average_inference_speed))

    write_speed(head, average_inference_speed)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.app.run(inference)
