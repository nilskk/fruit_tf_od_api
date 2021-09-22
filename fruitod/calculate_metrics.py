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
from absl import flags, app
from fruitod.utils import voc_util
from fruitod.utils import file_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

flags.DEFINE_string('model_path', None, 'Path to model (has checkpoints directory)')
flags.DEFINE_string('voc_path', None, 'Path to voc dataset folder with labelmap and tfrecord '
                                      'and optional weights and scaler file')
flags.DEFINE_string('labelmap_file', 'label_map.pbtxt', 'Name of labelmap File')
flags.DEFINE_string('set_file', 'test.txt', 'Name of file with list images to evaluate')
flags.DEFINE_string('scaler_file', 'minmax_scaler.pkl', 'Filename of scaler pickle')
flags.DEFINE_boolean('side_input', False, 'wether to include weights as side input into model')

flags.mark_flags_as_required(['model_path', 'voc_path'])
FLAGS = flags.FLAGS


def _read_annotations_for_groundtruth(voc_path, set_file, labelmap_dict):
    """
        Read Annotations from VOC Groundtruth.
    Args:
        voc_path: Path to used VOC dataset
        set_file: Path to of used set_file
        labelmap_dict: a dictionary mapping label names to id

    Returns:
        image_ids: List of unique identifiers for each image
        gt_boxes: A list of numpy arrays with size [NumberOfBoxes, 4] containing
            box coordinates [ymin, xmin, ymax, xmax]
        gt_classes: A list of numpy arrays with size [NumberOfBoxes, 1] containing class ids for each box

    """
    examples_path = os.path.join(voc_path, 'ImageSets', set_file)
    examples_list = dataset_util.read_examples_list(examples_path)

    image_ids = []
    gt_boxes = []
    gt_classes = []

    for example in examples_list:
        example_without_extension = Path(example).stem
        annotation_path = os.path.join(voc_path, 'Annotations', example_without_extension + '.xml')
        id, groundtruth_boxes, groundtruth_classes = voc_util.read_voc_annotation_file(annotation_path, labelmap_dict)
        image_ids.append(id)
        gt_boxes.append(groundtruth_boxes)
        gt_classes.append(groundtruth_classes)

    return image_ids, gt_boxes, gt_classes



def inference(model,
              voc_path,
              model_path,
              set_file,
              category_index,
              scaler_file=None,
              side_input=False):
    """
        Measure mean time of inference for number of images
        and plot detected boxes with class and optional weight
        on images.

    Args:
        model: Loaded SavedModel for inference
        voc_path: Path to used VOC dataset
        model_path: Path to model directory
        set_file: Path to of used set_file
        category_index: a category index, which is a dictionary that maps integer ids to dicts containing categories
        scaler_file: Path to used scaler pickle file, if weights are used as input
        side_input: True if weights are used as input

    Returns:
        dt_ids: List of numpy arrays with size [NumberOfBoxes, 1] containing the unique identifier of the image
        dt_boxes: List of numpy arrays with size [NumberOfBoxes, 4] containing the box coordinates
            [ymin, xmin, ymax, xmax] for each box
        dt_classes: List of numpy arrays with size [NumberOfBoxes] containing the detected class index of each box
        dt_scores: List of numpy arrays with size [NumberOfBoxes] containing the class score of each box
        mean_elapsed: Float of mean elapsed time in milliseconds for inference
    """
    examples_path = os.path.join(voc_path, 'ImageSets', set_file)
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

        # if use weights, then scale weights with saved scaler pickle file from create_tfrecord_from_voc.py
        if side_input:
            scaler = pickle.load(open(os.path.join(voc_path, scaler_file), 'rb'))
            weight_path = os.path.join(voc_path, 'Weights', example_without_extension + '.json')
            with open(weight_path) as f:
                json_dict = json.load(f)
            weight = json_dict['weightInGrams']
            weight_scaled = scaler.transform(np.asarray(weight).reshape(-1, 1))
            weight_scaled = np.asarray([np.squeeze(weight_scaled)], dtype=np.float32)

        image_path = os.path.join(voc_path, 'JPEGImages', example)
        image_np = file_util.load_image_into_numpy_array(image_path)
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

        # Additional class agnostic NMS processing to select best box of object for multi label classification
        indices = tf.image.non_max_suppression(boxes=detection_boxes,
                                               scores=detection_scores,
                                               iou_threshold=0.95,
                                               max_output_size=64)
        indices_np = indices.numpy()

        detection_weights = None
        if 'detection_weightPerObject' in detections:
            detection_weights = detections['detection_weightPerObject'][0].numpy()
            detection_weights = detection_weights[indices_np]

        image_np_with_detections = image_np.copy()
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=detection_boxes[indices_np],
            classes=detection_classes[indices_np],
            scores=detection_scores[indices_np],
            weights=detection_weights,
            category_index=category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=64,
            min_score_thresh=0.5,
            line_thickness=2
        )
        im = Image.fromarray(image_np_with_detections)
        png_path = Path(os.path.join(model_path, 'metrics', 'detections'))
        png_path.mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(png_path, example))

    mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

    return dt_ids, dt_boxes, dt_classes, dt_scores, mean_elapsed


def get_flops(model):
    """
        Calculate Flops of model with Tensorflow 1 Profiler
    Args:
        model: Loaded SavedModel for inference

    Returns:
        total_flops: Integer with Number of total FLOPs of model

    """
    full_model = tf.function(lambda x: model(x))

    full_model = full_model.get_concrete_function(
        tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8, name='input_tensor'))

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        print("Flops: {}".format(flops.total_float_ops))
        total_flops = flops.total_float_ops

    return total_flops


def main(_):
    model_path = FLAGS.model_path
    voc_path = FLAGS.voc_path
    labelmap_file = FLAGS.labelmap_file
    set_file = FLAGS.set_file
    scaler_file = FLAGS.scaler_file
    side_input = FLAGS.side_input

    # Load Model and read label_map.pbtxt
    model = file_util.load_model(model_path)
    categories, labelmap_dict, category_index = file_util.load_labelmap(voc_path, labelmap_file)

    # Get information from groundtruth and detection
    gt_ids, gt_boxes, gt_classes = _read_annotations_for_groundtruth(voc_path, set_file, labelmap_dict)
    dt_ids, dt_boxes, dt_classes, dt_scores, time_per_image = inference(model,
                                                                        voc_path,
                                                                        model_path,
                                                                        set_file,
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

    # Convert to Percent format
    for k, v in summary_metrics.items():
        summary_metrics[k] = v * 100
    for k, v in per_category_ap.items():
        per_category_ap[k] = v * 100
    print(summary_metrics)
    print(per_category_ap)

    # Prevention for Tensorflow Bug: Cant calculate Flops with custom Inputs
    if not side_input:
        flops = get_flops(model)
    else:
        flops = 0

    metrics_dict = {'flops': flops / 1e9,
                    'time_per_image': time_per_image * 1000}

    # Read Trainable Params and Name Dictionary from Pickle file
    name_params_dict = pickle.load(open(os.path.join(model_path, 'metrics', 'name_params.pkl'), 'rb'))
    metrics_dict.update(name_params_dict)

    metrics_dict.update(summary_metrics)
    metrics_dict.update(per_category_ap)

    # Save Metrics to CSV
    metrics_df = pd.DataFrame.from_records([metrics_dict])
    metrics_df.to_csv(os.path.join(model_path, 'metrics', 'metrics.csv'))


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    app.run(main)
