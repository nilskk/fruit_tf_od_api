import os
import json
import pickle
import time
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
from six import BytesIO
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils
from object_detection.metrics import coco_tools
from lxml import etree
from fruitod.utils.file_util import read_tfrecord
from absl import flags, app
from fruitod.utils import voc_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

flags.DEFINE_string('model_path', None, 'Path to model')
flags.DEFINE_string('data_path', None, 'Path to dataset')
flags.DEFINE_string('tfrecord_file', 'test_gpu_0.tfrecord', 'Name of tfrecord File')
flags.DEFINE_string('labelmap_file', 'pascal_label_map.pbtxt', 'Name of labelmap File')
flags.DEFINE_string('examples_file', 'test.txt', 'Name of file with test images')
flags.DEFINE_string('scaler_file', None, 'Filename of scaler pickle')
flags.DEFINE_boolean('side_input', False, 'wether to include weights as side input into model')

flags.mark_flags_as_required(['model_path', 'data_path'])
FLAGS = flags.FLAGS

def _load_model(model_path):
    saved_model_path = os.path.join(model_path, 'export', 'saved_model')
    model = tf.saved_model.load(saved_model_path)
    infer = model.signatures["serving_default"]
    print(infer)
    return model

def _load_labelmap(data_path, labelmap_file):

    labelmap_path = os.path.join(data_path, labelmap_file)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path)
    categories = label_map_util.create_categories_from_labelmap(labelmap_path)
    labelmap_dict = label_map_util.get_label_map_dict(labelmap_path)

    return categories, labelmap_dict, category_index

def _read_annotations_for_groundtruth(data_path, examples_file, labelmap_dict):
    examples_path = os.path.join(data_path, 'ImageSets', examples_file)
    examples_list = dataset_util.read_examples_list(examples_path)

    image_ids = []
    gt_boxes = []
    gt_classes = []

    for example in examples_list:
        example_without_extension = Path(example).stem
        annotation_path = os.path.join(data_path, 'Annotations', example_without_extension + '.xml')
        id, groundtruth_boxes, groundtruth_classes = voc_util.read_voc_for_groundtruth(annotation_path, labelmap_dict)
        image_ids.append(id)
        gt_boxes.append(groundtruth_boxes)
        gt_classes.append(groundtruth_classes)

    return image_ids, gt_boxes, gt_classes

# From Tensorflow Object Detection API object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb
def _load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def inference(model,
              data_path,
              model_path,
              examples_file,
              category_index,
              scaler_file=None,
              side_input=False):
    examples_path = os.path.join(data_path, 'ImageSets', examples_file)
    examples_list = dataset_util.read_examples_list(examples_path)

    dt_classes = []
    dt_boxes = []
    dt_scores = []
    dt_ids = []

    elapsed_time = []

    print(side_input)

    for example in examples_list:
        example_without_extension = Path(example).stem

        dt_ids.append(example)

        if side_input:
            scaler = pickle.load(open(os.path.join(data_path, scaler_file), 'rb'))
            weight_path = os.path.join(data_path, 'Weights', example_without_extension + '.json')
            with open(weight_path) as f:
                json_dict = json.load(f)
            weight = json_dict['weightInGrams']
            weight_scaled = scaler.transform(np.asarray(weight).reshape(-1, 1))
            weight_scaled =np.asarray([np.squeeze(weight_scaled)], dtype=np.float32)

        image_path = os.path.join(data_path, 'JPEGImages', example)
        image_np = _load_image_into_numpy_array(image_path)
        image_height = image_np.shape[0]
        image_width = image_np.shape[1]
        input_tensor = np.expand_dims(image_np, 0)

        if side_input:
            start = time.time()
            detections = model(input_tensor, weight_scaled)
            end = time.time()
        else:
            start = time.time()
            detections = model(input_tensor)
            end = time.time()
        elapsed_time.append(end - start)

        shape_array = np.array([image_height, image_width, image_height, image_width])

        detection_scores = detections['detection_scores'][0].numpy()
        detection_boxes = detections['detection_boxes'][0].numpy() * shape_array[np.newaxis, :]
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)

        dt_boxes.append(detection_boxes)
        dt_classes.append(detection_classes)
        dt_scores.append(detection_scores)

        detection_weights = None
        if 'detection_weightPerObject' in detections:
            detection_weights = detections['detection_weightPerObject']

        image_np_with_detections = image_np.copy()
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=detection_boxes,
            classes=detection_classes,
            scores=detection_scores,
            weights=detection_weights,
            category_index=category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=100,
            min_score_thresh=0.5,
            line_thickness=2
        )
        im = Image.fromarray(image_np_with_detections)
        png_path = Path(os.path.join(model_path, 'inference', 'detections'))
        png_path.mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(png_path, example))

    mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

    return dt_ids, dt_boxes, dt_classes, dt_scores, mean_elapsed

def get_flops(model, side_input=False):
    full_model = tf.function(lambda x: model(x))
    if side_input:
        full_model = full_model.get_concrete_function([tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8),
                                                      tf.TensorSpec(shape=[1], dtype=tf.float32)])
    else:
        full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        print("Flops: {}".format(flops.total_float_ops))

    return flops.total_float_ops



def main(_):
    model_path = FLAGS.model_path
    data_path = FLAGS.data_path
    tfrecord_file = FLAGS.tfrecord_file
    labelmap_file = FLAGS.labelmap_file
    examples_file = FLAGS.examples_file
    scaler_file = FLAGS.scaler_file
    side_input = FLAGS.side_input

    model = _load_model(model_path)
    categories, labelmap_dict, category_index = _load_labelmap(data_path, labelmap_file)
    gt_ids, gt_boxes, gt_classes = _read_annotations_for_groundtruth(data_path, examples_file, labelmap_dict)
    dt_ids, dt_boxes, dt_classes, dt_scores, time_per_image = inference(model,
                                                        data_path,
                                                        model_path,
                                                        examples_file,
                                                        category_index,
                                                        scaler_file,
                                                        side_input)

    # COCO Evaluation
    groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(gt_ids, gt_boxes, gt_classes, categories)
    detections_list = coco_tools.ExportDetectionsToCOCO(dt_ids, dt_boxes, dt_scores, dt_classes, categories)
    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, agnostic_mode=False)
    summary_metrics, per_category_ap = evaluator.ComputeMetrics(include_metrics_per_category=True,
                                                                all_metrics_per_category=True)
    for k, v in summary_metrics.items():
        summary_metrics[k] = round(v*100, 2)
    for k, v in per_category_ap.items():
        per_category_ap[k] = round(v*100, 2)
    print(summary_metrics)
    print(per_category_ap)

    flops = get_flops(model, side_input)

    metrics_dict = {'flops': round(flops / 1e9, 2),
                    'time_per_image': round(time_per_image*1000, 2)}

    metrics_dict.update(summary_metrics)
    metrics_dict.update(per_category_ap)

    metrics_df = pd.DataFrame.from_records([metrics_dict])
    metrics_df.to_csv(os.path.join(model_path, 'inference', 'metrics.csv'))


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    app.run(main)
