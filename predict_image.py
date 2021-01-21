from absl import flags, app
import os
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import pandas as pd

flags.DEFINE_string('export_dir', None, 'Path to exported model')
flags.DEFINE_string('label_map_path', './data/voc_data/pascal_label_map.pbtxt',
                    'Path to label map proto')

flags.DEFINE_float('threshold', 0.5, 'Minimum score threshold')
flags.DEFINE_string('image', None, 'Path to single image')
flags.DEFINE_string('image_folder', None, 'Path to image folder')
flags.DEFINE_boolean('visualize', True, 'Visualize Object Detection results')
flags.DEFINE_string('output', None, 'Path for output files')

FLAGS = flags.FLAGS


def load_image_as_nparray(path):
    return np.array(Image.open(path))


def visualize_image(model, image_path, categories):
    image_name = Path(image_path).stem
    image_np = load_image_as_nparray(image_path)
    image_tensor = tf.convert_to_tensor(image_np)
    image_tensor = image_tensor[tf.newaxis, ...]
    tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

    detections = model(image_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    detection_scores_threshold = np.expand_dims(detections['detection_scores'][detections['detection_scores'] > FLAGS.threshold], axis=0)
    detection_classes_threshold = np.expand_dims(detections['detection_classes'][detections['detection_scores'] > FLAGS.threshold], axis=0)
    detection_boxes_threshold = np.swapaxes(detections['detection_boxes'][detections['detection_scores'] > FLAGS.threshold], axis1=0, axis2=1)
    detection_matrix = np.concatenate((np.concatenate((detection_classes_threshold, detection_scores_threshold), axis=0), detection_boxes_threshold), axis=0)

    detection_frame = pd.DataFrame({'class': detection_matrix[0, :], 'score': detection_matrix[1, :],
                                    'bbox_x_min': detection_matrix[2, :], 'bbox_y_min': detection_matrix[3, :],
                                    'bbox_x_max': detection_matrix[4, :], 'bbox_y_max': detection_matrix[5, :]})

    csv_path = Path(os.path.join(FLAGS.output, 'csv'))
    csv_path.mkdir(parents=True, exist_ok=True)
    detection_frame.to_csv(os.path.join(csv_path, image_name + '.csv'), index=False)
    
    if FLAGS.visualize is True:
        image_np_with_detections = image_np.copy()

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=detections['detection_boxes'],
            classes=detections['detection_classes'],
            scores=detections['detection_scores'],
            category_index=categories,
            use_normalized_coordinates=True,
            min_score_thresh=FLAGS.threshold
        )

        im = Image.fromarray(image_np_with_detections)
        png_path = Path(os.path.join(FLAGS.output, 'plots'))
        png_path.mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(png_path, image_name + '.png'))


def predict(argv):
    flags.mark_flag_as_required('export_dir')
    flags.mark_flag_as_required('output')
    flags.mark_flags_as_mutual_exclusive(flag_names=['image', 'image_folder'], required=True)

    saved_model_path = os.path.join(FLAGS.export_dir, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    categories = label_map_util.create_category_index_from_labelmap(FLAGS.label_map_path, use_display_name=True)

    if FLAGS.image_folder is not None:
        for image_name in os.listdir(FLAGS.image_folder):
            image_path = os.path.join(FLAGS.image_folder, image_name)
            visualize_image(model=model, image_path=image_path, categories=categories)
    elif FLAGS.image is not None:
        image_path = FLAGS.image
        visualize_image(model=model, image_path=image_path, categories=categories)


if __name__ == '__main__':
    app.run(predict)
