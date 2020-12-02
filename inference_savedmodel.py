import time
import os
import math
import random
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from csv_util import write_speed
from absl import flags

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

    categories = label_map_util.create_category_index_from_labelmap(FLAGS.label_map_path, use_display_name=True)

    image_dir = "./data/test_data/images"
    num_images = 16

    time_diffs = []

    rows = 4
    cols = int(math.ceil(num_images / rows))
    gs = plt.GridSpec(rows, cols)
    fig = plt.figure(figsize=(20, 20))
    fig.tight_layout()

    for i, image_name in enumerate(random.sample(os.listdir(image_dir), k=num_images)):
        image_path = os.path.join(image_dir, image_name)
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

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=detections['detection_boxes'],
            classes=detections['detection_classes'],
            scores=detections['detection_scores'],
            category_index=categories,
            use_normalized_coordinates=True,
            min_score_thresh=0.5
        )

        ax = fig.add_subplot(gs[i])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(image_np_with_detections)

    head, tail = os.path.split(str(FLAGS.export_dir))

    png_path = os.path.join(head, 'result.png')
    plt.savefig(png_path, bbox_inches='tight')

    time_diffs_array = np.array(time_diffs)

    average_inference_speed = np.average(time_diffs_array[1:])
    print("Average Inference Speed: " + str(average_inference_speed))

    write_speed(head, average_inference_speed)


if __name__ == "__main__":
    tf.compat.v1.app.run(inference)
