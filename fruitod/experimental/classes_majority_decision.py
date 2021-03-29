from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.utils import label_map_util
from fruitod.utils.voc_util import read_voc_for_detections

from pathlib import Path
from fruitod.settings import *
from lxml import etree
import numpy as np
import pandas as pd


if __name__ == '__main__':
    groundtruth_annotation_path = os.path.join(VOC_PATH, 'Annotations')
    prediction_annotation_path = os.path.join(PREDICTION_OUTPUT_PATH, 'voc/Annotations')

    category_name_to_index = label_map_util.get_label_map_dict(LABELMAP_PATH)
    categories = label_map_util.create_categories_from_labelmap(LABELMAP_PATH)
    evaluator = CocoDetectionEvaluator(categories=categories,
                                       include_metrics_per_category=True,
                                       all_metrics_per_category=False)

    for file in os.listdir(prediction_annotation_path):
        #add groundtruth from voc file to evaluator
        image_id, groundtruth_boxes, groundtruth_classes, groundtruth_scores = read_voc_for_detections(
            os.path.join(groundtruth_annotation_path, file))
        groundtruth_classes_index = np.vectorize(category_name_to_index.get)(groundtruth_classes)
        groundtruth = {'groundtruth_boxes': groundtruth_boxes,
                       'groundtruth_classes': groundtruth_classes_index}
        evaluator.add_single_ground_truth_image_info(image_id=image_id, groundtruth_dict=groundtruth)

        #add detections from voc file to evaluator
        image_id, detection_boxes, detection_classes, detection_scores = read_voc_for_detections(
            os.path.join(prediction_annotation_path, file), scores=True)
        detection_classes_index = np.vectorize(category_name_to_index.get)(detection_classes)
        detections = {'detection_boxes': detection_boxes,
                      'detection_classes': detection_classes_index,
                      'detection_scores': detection_scores}
        evaluator.add_single_detected_image_info(image_id=image_id, detections_dict=detections)

    metrics = evaluator.evaluate()
    dataframe = pd.DataFrame.from_records(metrics, index=[0])
    dataframe.to_csv(os.path.join(PREDICTION_OUTPUT_PATH, 'comparison.csv'), index=False)



