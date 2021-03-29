from lxml import etree
import numpy as np
from object_detection.utils import label_map_util


def read_voc_for_detections(voc_file_path, scores=False):
    root = etree.parse(voc_file_path)
    image_id = root.find('filename').text
    objects = root.findall('object')

    detection_boxes = np.array([]).reshape(0, 4)
    detection_classes = np.array([])
    detection_scores = np.array([])

    for item in objects:
        name = item.find('name').text
        name_array = np.array([name])
        detection_classes = np.concatenate([detection_classes, name_array], axis=0)

        if scores:
            score = item.find('score').text
            score_array = np.array([score]).astype(np.float)
            detection_scores = np.concatenate([detection_scores, score_array], axis=0)

        bndbox = item.find('bndbox')
        ymin = bndbox.find('ymin').text
        xmin = bndbox.find('xmin').text
        ymax = bndbox.find('ymax').text
        xmax = bndbox.find('xmax').text
        bndbox_array = np.expand_dims(np.array([ymin, xmin, ymax, xmax]).astype(np.float), axis=0)
        detection_boxes = np.concatenate([detection_boxes, bndbox_array], axis=0)

    if detection_boxes.ndim < 2:
        detection_boxes = np.expand_dims(detection_boxes, axis=0)

    return image_id, detection_boxes, detection_classes, detection_scores
        

