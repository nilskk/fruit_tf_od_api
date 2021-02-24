from absl import flags, app
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np
import io
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import pandas as pd
from lxml import etree as ET
from utils.file_util import read_tfrecord

flags.DEFINE_string('export_path', None, 'Path to exported model')
flags.DEFINE_string('output_path', None, 'Path for output files')
flags.DEFINE_string('tfrecord_path', None, 'Path to tfrecord file')
flags.DEFINE_string('labelmap_path', './data/voc_data/pascal_label_map.pbtxt',
                    'Path to label map proto')

flags.DEFINE_float('score_threshold', 0.7, 'Minimum score threshold')
flags.DEFINE_float('iou_threshold', 0.95, 'Minimum iou threshold')
flags.DEFINE_boolean('visualize', True, 'Visualize Object Detection results')


FLAGS = flags.FLAGS


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

    root = ET.Element('annotation')
    root.set('verified', 'yes')

    folder = ET.SubElement(root, 'folder')
    folder.text = 'Annotation'

    filename = ET.SubElement(root, 'filename')
    filename.text = image_name

    path = ET.SubElement(root, 'path')
    path.text = os.path.join(voc_image_path, image_name)

    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(im.width)
    height = ET.SubElement(size, 'height')
    height.text = str(im.height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = str(0)

    for row, category in zip(detections['detection_boxes'], detections['detection_classes']):
        object = ET.SubElement(root, 'object')

        name = ET.SubElement(object, 'name')
        name.text = categories[category]['name']
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = str(0)

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(row[1]*im.width)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(row[0]*im.height)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(row[3]*im.width)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(row[2]*im.height)

    xml_string = ET.tostring(root, pretty_print=True)
    with open(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), 'wb') as files:
        files.write(xml_string)


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


def main(argv):
    flags.mark_flag_as_required('export_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('tfrecord_path')

    predict(export_path=FLAGS.export_path,
            output_path=FLAGS.output_path,
            labelmap_path=FLAGS.labelmap_path,
            tfrecord_path=FLAGS.tfrecord_path,
            score_threshold=FLAGS.score_threshold,
            iou_threshold=FLAGS.iou_threshold,
            visualize=FLAGS.visualize)


if __name__ == '__main__':
    app.run(main)
