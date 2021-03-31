import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
from lxml import etree
from fruitod.utils.file_util import read_tfrecord
from fruitod.settings import *


def filter_detections_nms(detections, iou_threshold, score_threshold):
    indices = tf.image.non_max_suppression(boxes=detections['detection_boxes'], scores=detections['detection_scores'],
                                           iou_threshold=iou_threshold, score_threshold=score_threshold, max_output_size=30)

    return indices.numpy()


def calculate_detections(model,
                         image,
                         score_threshold,
                         iou_threshold):

    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]
    tf.image.convert_image_dtype(image, dtype=tf.uint8)

    detections = model(image_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    indices = filter_detections_nms(detections, iou_threshold=iou_threshold,
                                    score_threshold=score_threshold)

    detections_with_nms_filter = {'detection_boxes': detections['detection_boxes'][indices],
                                  'detection_classes': detections['detection_classes'][indices],
                                  'detection_scores': detections['detection_scores'][indices]}

    return detections_with_nms_filter


def visualize_detections(output_path,
                         image,
                         image_name,
                         categories,
                         detections):

    image_np_with_detections = image.copy()

    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes=detections['detection_boxes'],
        classes=detections['detection_classes'],
        scores=detections['detection_scores'],
        category_index=categories,
        use_normalized_coordinates=True
    )

    im = Image.fromarray(image_np_with_detections)
    png_path = Path(os.path.join(output_path, 'plots'))
    png_path.mkdir(parents=True, exist_ok=True)
    im.save(os.path.join(png_path, image_name))


def create_voc(output_path,
               image,
               image_name,
               categories,
               detections):
    voc_annotation_path = Path('voc/Annotations')
    complete_voc_annotation_path = Path(os.path.join(output_path, voc_annotation_path))
    complete_voc_annotation_path.mkdir(parents=True, exist_ok=True)

    voc_image_path = Path('voc/JPEGImages')
    complete_voc_image_path = Path(os.path.join(output_path, voc_image_path))
    complete_voc_image_path.mkdir(parents=True, exist_ok=True)

    image_name_without_extension = Path(image_name).stem

    im = Image.fromarray(image)
    im.save(os.path.join(complete_voc_image_path, image_name))

    root = etree.Element('annotation')
    root.set('verified', 'yes')

    folder = etree.SubElement(root, 'folder')
    folder.text = 'Annotation'

    filename = etree.SubElement(root, 'filename')
    filename.text = image_name

    path = etree.SubElement(root, 'path')
    path.text = os.path.join(voc_image_path, image_name)

    source = etree.SubElement(root, 'source')
    database = etree.SubElement(source, 'database')
    database.text = 'Unknown'

    size = etree.SubElement(root, 'size')
    width = etree.SubElement(size, 'width')
    width.text = str(im.width)
    height = etree.SubElement(size, 'height')
    height.text = str(im.height)
    depth = etree.SubElement(size, 'depth')
    depth.text = str(3)

    segmented = etree.SubElement(root, 'segmented')
    segmented.text = str(0)

    for row, category, detection_score in zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
        object = etree.SubElement(root, 'object')

        name = etree.SubElement(object, 'name')
        name.text = categories[category]['name']
        pose = etree.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = etree.SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = etree.SubElement(object, 'difficult')
        difficult.text = str(0)

        score = etree.SubElement(object, 'score')
        score.text = str(detection_score)

        bndbox = etree.SubElement(object, 'bndbox')
        xmin = etree.SubElement(bndbox, 'xmin')
        xmin.text = str(row[1]*im.width)
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = str(row[0]*im.height)
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = str(row[3]*im.width)
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = str(row[2]*im.height)

    # xml_string = etree.tostring(root, pretty_print=True)
    # with open(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), 'wb') as files:
    #     files.write(xml_string)
    tree = etree.ElementTree(root)
    tree.write(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), pretty_print=True)


def predict(export_path,
            output_path,
            labelmap_path,
            tfrecord_path,
            score_threshold=0.5,
            iou_threshold=0.5,
            visualize=False):
    tf.config.set_visible_devices([], 'GPU')

    saved_model_path = os.path.join(export_path, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    categories = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    records = read_tfrecord(tfrecord_path)
    for record in records:
        detections = calculate_detections(model=model,
                                          image=record['image'],
                                          score_threshold=score_threshold,
                                          iou_threshold=iou_threshold)

        if visualize:
            visualize_detections(image=record['image'],
                                 image_name=record['filename'],
                                 output_path=output_path,
                                 categories=categories,
                                 detections=detections)

        create_voc(image=record['image'],
                   image_name=record['filename'],
                   output_path=output_path,
                   categories=categories,
                   detections=detections)


def main():
    predict(export_path=EXPORT_PATH,
            output_path=PREDICTION_OUTPUT_PATH,
            labelmap_path=LABELMAP_PATH,
            tfrecord_path=VAL_TFRECORD_PATH,
            score_threshold=SCORE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            visualize=VISUALIZE)


if __name__ == '__main__':
    main()
