from lxml import etree
import numpy as np
from object_detection.utils import label_map_util

def read_voc_for_detections(voc_file_path):
    root = etree.parse(voc_file_path)
    objects = root.findall('object')
    
    detection_boxes = np.array([])
    detection_classes = np.array([])
    detection_scores = np.array([])

    for idx, item in enumerate(objects):
        score = item.find('score').text
        score_array = np.array([score]).astype(np.float)
        
        id = item.find('id').text
        id_array = np.array([id]).astype(np.int64)

        bndbox = item.find('bndbox')
        ymin = bndbox.find('ymin').text
        xmin = bndbox.find('xmin').text
        ymax = bndbox.find('ymax').text
        xmax = bndbox.find('xmax').text
        bndbox_array = np.array([ymin, xmin, ymax, xmax]).astype(np.float)
        
        detection_scores = np.concatenate([detection_scores, score_array], axis=0)
        detection_classes = np.concatenate([detection_classes, id_array], axis=0)
        detection_boxes = np.concatenate([detection_boxes, bndbox_array], axis=0)

    print(detection_boxes.shape)
    detections = {'detection_boxes': detection_boxes,
                  'detection_scores': detection_scores,
                  'detection_classes': detection_classes}

    return detections
        

