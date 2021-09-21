from lxml import etree
import numpy as np
from object_detection.utils import label_map_util


def read_voc_annotation_file(annotation_file_path, labelmap_dict, has_scores=False):
    """
        Read VOC Annotation XML file with optional scores
    Args:
        annotation_file_path: Path to VOC Annotation XML file
        labelmap_dict: a dictionary mapping label names to id
        has_scores: True, if Annotation file has field 'score' to read

    Returns:
        image_id: Unique identifier of image
        boxes: Numpy array of size [NumberOfBoxes, 4] containing box coordinates [ymin, xmin, ymax, xmax]
        classes: Numpy array of size [NumberOfBoxes, 1] containing class index for each box
        scores: (optional) Numpy array of size [NumberOfBoxes, 1] containing scores of detected boxes

    """
    root = etree.parse(annotation_file_path)
    image_id = root.find('filename').text
    objects = root.findall('object')

    boxes = np.array([]).reshape(0, 4)
    classes = np.array([])
    scores = np.array([])

    for item in objects:
        name = item.find('name').text
        class_id = labelmap_dict[name]
        class_array = np.array([class_id])
        classes = np.concatenate([classes, class_array], axis=0)

        if has_scores:
            score = item.find('score').text
            score_array = np.array([score]).astype(np.float)
            scores = np.concatenate([scores, score_array], axis=0)

        bndbox = item.find('bndbox')
        ymin = bndbox.find('ymin').text
        xmin = bndbox.find('xmin').text
        ymax = bndbox.find('ymax').text
        xmax = bndbox.find('xmax').text
        bndbox_array = np.expand_dims(np.array([ymin, xmin, ymax, xmax]).astype(np.float32), axis=0)
        boxes = np.concatenate([boxes, bndbox_array], axis=0)

    if boxes.ndim < 2:
        boxes = np.expand_dims(boxes, axis=0)

    classes = classes.astype(np.int32)

    if has_scores:
        return image_id, boxes, classes, scores
    else:
        return image_id, boxes, classes


def write_class_name(annotation_file_path, class_name):
    """
        Write class name to each object in VOC Annotation XML file
    Args:
        annotation_file_path: Path to VOC Annotation XML file
        class_name: String containing the name of the class
    """
    root = etree.parse(annotation_file_path)
    objects = root.findall('object')
    
    for item in objects:
        name = item.find('name')
        name.text = class_name

    root.write(annotation_file_path, pretty_print=True)
        

