from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.utils import label_map_util
from fruitod.utils.voc_util import read_voc_for_detections, write_class_name

from pathlib import Path
from fruitod.settings_gpu_0 import *
from lxml import etree
import numpy as np
import pandas as pd

def soft_voting_decision(detection_classes_id, detection_scores):
    id_to_category_dict = label_map_util.create_category_index_from_labelmap(LABELMAP_PATH)

    values, ids, counts = np.unique(detection_classes_id, return_inverse=True, return_counts=True)
    weights = counts / detection_classes_id.size
    mean_propabilities = np.bincount(ids, detection_scores) / counts

    best_class_id = values[np.argmax(weights * mean_propabilities)]

    best_class_name = id_to_category_dict[best_class_id]['name']

    return best_class_name



if __name__ == '__main__':
    groundtruth_annotation_path = os.path.join(VOC_PATH, 'Annotations')
    prediction_annotation_path = os.path.join(PREDICTION_OUTPUT_PATH, 'voc/Annotations')
    softvoting_path = os.path.join(PREDICTION_OUTPUT_PATH, 'voc/Annotations_softvoting')

    category_name_to_index = label_map_util.get_label_map_dict(LABELMAP_PATH)
    id_to_category_list = label_map_util.create_categories_from_labelmap(LABELMAP_PATH)

    evaluator_base = CocoDetectionEvaluator(categories=id_to_category_list,
                                            include_metrics_per_category=True,
                                            all_metrics_per_category=False)
    evaluator_softvoting = CocoDetectionEvaluator(categories=id_to_category_list,
                                                include_metrics_per_category=True,
                                                all_metrics_per_category=False)

    for file in os.listdir(prediction_annotation_path):
        # add groundtruth from voc file to evaluator_base and evaluator_softvoting
        groundtruth_image_id, groundtruth_boxes, groundtruth_classes, groundtruth_scores = read_voc_for_detections(
            os.path.join(groundtruth_annotation_path, file))
        if groundtruth_classes.size == 0:
            groundtruth_classes_index = np.empty(groundtruth_classes.shape)
        else:
            groundtruth_classes_index = np.vectorize(category_name_to_index.get)(groundtruth_classes)
        groundtruth = {'groundtruth_boxes': groundtruth_boxes,
                       'groundtruth_classes': groundtruth_classes_index}
        evaluator_base.add_single_ground_truth_image_info(image_id=groundtruth_image_id, groundtruth_dict=groundtruth)
        evaluator_softvoting.add_single_ground_truth_image_info(
            image_id=groundtruth_image_id, groundtruth_dict=groundtruth)

        # add detections from voc file to evaluator
        detection_image_id, detection_boxes, detection_classes, detection_scores = read_voc_for_detections(
            os.path.join(prediction_annotation_path, file), scores=True)
        if detection_classes.size == 0:
            detection_classes_index = np.empty(detection_classes.shape)
        else:
            detection_classes_index = np.vectorize(category_name_to_index.get)(detection_classes)
        detections = {'detection_boxes': detection_boxes,
                      'detection_classes': detection_classes_index,
                      'detection_scores': detection_scores}
        evaluator_base.add_single_detected_image_info(image_id=detection_image_id, detections_dict=detections)

        #get class_name from soft voting majority and write in xml
        if detection_classes_index.size != 0:
            best_class_name = soft_voting_decision(detection_classes_index, detection_scores)
            write_class_name(os.path.join(softvoting_path, file), best_class_name)

        # add detections from voc file to evaluator
        detection_image_id, detection_boxes, detection_classes, detection_scores = read_voc_for_detections(
            os.path.join(prediction_annotation_path, file), scores=True)
        if detection_classes.size == 0:
            detection_classes_index = np.empty(detection_classes.shape)
        else:
            detection_classes_index = np.vectorize(category_name_to_index.get)(detection_classes)
        detections = {'detection_boxes': detection_boxes,
                      'detection_classes': detection_classes_index,
                      'detection_scores': detection_scores}
        evaluator_softvoting.add_single_detected_image_info(image_id=detection_image_id, detections_dict=detections)

    metrics_base = evaluator_base.evaluate()
    dataframe_base = pd.DataFrame.from_records(metrics_base, columns=metrics_base.keys(), index=['base'])

    metrics_softvoting = evaluator_softvoting.evaluate()
    dataframe_softvoting = \
        pd.DataFrame.from_records(metrics_softvoting, columns=metrics_softvoting.keys(), index=['softvoting'])

    dataframe_ges = pd.concat([dataframe_base, dataframe_softvoting])
    dataframe_ges.to_csv(os.path.join(PREDICTION_OUTPUT_PATH, 'comparison_base_softvoting.csv'), index=True)



