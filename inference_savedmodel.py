import time
import os
import math
import random
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

LABELMAP_PATH="./data/voc_data/pascal_label_map.pbtxt"
MODEL_PATH = "./models/exported_models/mobilenetv2_fpnlite_lr=0.04"
SAVED_MODEL_PATH = MODEL_PATH + "/saved_model"


def load_image_as_nparray(path):
    return np.array(Image.open(path))


if __name__=="__main__":

    model = tf.saved_model.load(SAVED_MODEL_PATH)

    categories = label_map_util.create_category_index_from_labelmap(LABELMAP_PATH, use_display_name=True)

    image_dir="./data/test_data"
    num_images = 9

    time_diffs = []

    rows = 3
    cols = int(math.ceil(num_images/rows))
    gs = plt.GridSpec(rows, cols)
    fig = plt.figure()
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

        #detections['detection_classes'] = tf.cast(detections['detection_classes'], dtype=tf.int64)

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
            min_score_thresh=0.9
        )

        ax = fig.add_subplot(gs[i])
        ax.imshow(image_np_with_detections)

    plt.show()

    time_diffs_array = np.array(time_diffs)

    print("Average Inference Speed: " + str(np.average(time_diffs_array[1:])))






