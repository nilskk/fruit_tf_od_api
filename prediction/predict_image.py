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

flags.DEFINE_string('export_dir', None, 'Path to exported model')
flags.DEFINE_string('label_map_path', './data/voc_data/pascal_label_map.pbtxt',
                    'Path to label map proto')

flags.DEFINE_float('score_threshold', 0.7, 'Minimum score threshold')
flags.DEFINE_float('iou_threshold', 0.95, 'Minimum iou threshold')
flags.DEFINE_string('image', None, 'Path to single image')
flags.DEFINE_string('image_folder', None, 'Path to image folder')
flags.DEFINE_string('tfrecord', None, 'Path to tfrecord file')
flags.DEFINE_boolean('visualize', True, 'Visualize Object Detection results')
flags.DEFINE_string('output', None, 'Path for output files')

FLAGS = flags.FLAGS

def filter_detections_nms(detections, iou_threshold, score_threshold):
    indices = tf.image.non_max_suppression(boxes=detections['detection_boxes'], scores=detections['detection_scores'],
                                           iou_threshold=iou_threshold, score_threshold=score_threshold, max_output_size=30)

    return indices.numpy()


def visualize_image(model, image, image_name, categories):
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

    indices = filter_detections_nms(detections, iou_threshold=FLAGS.iou_threshold, score_threshold=FLAGS.score_threshold)

    detection_scores_threshold = np.expand_dims(detections['detection_scores'][indices], axis=0)
    detection_classes_threshold = np.expand_dims(detections['detection_classes'][indices], axis=0)
    detection_boxes_threshold = np.swapaxes(detections['detection_boxes'][indices], axis1=0, axis2=1)
    detection_matrix = np.concatenate((np.concatenate((detection_classes_threshold, detection_scores_threshold), axis=0), detection_boxes_threshold), axis=0)

    # detection_frame = pd.DataFrame({'class': detection_matrix[0, :], 'score': detection_matrix[1, :],
    #                                 'bbox_x_min': detection_matrix[2, :], 'bbox_y_min': detection_matrix[3, :],
    #                                 'bbox_x_max': detection_matrix[4, :], 'bbox_y_max': detection_matrix[5, :]})
    #
    # csv_path = Path(os.path.join(FLAGS.output, 'csv'))
    # csv_path.mkdir(parents=True, exist_ok=True)
    # detection_frame.to_csv(os.path.join(csv_path, image_name + '.csv'), index=False)
    
    if FLAGS.visualize is True:
        image_np_with_detections = image.copy()

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=detections['detection_boxes'][indices],
            classes=detections['detection_classes'][indices],
            scores=detections['detection_scores'][indices],
            category_index=categories,
            use_normalized_coordinates=True
        )

        im = Image.fromarray(image_np_with_detections)
        png_path = Path(os.path.join(FLAGS.output, 'plots'))
        png_path.mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(png_path, image_name))

        create_voc(image_name=image_name, image=image, detection_matrix=detection_matrix, categories=categories)


def create_voc(image_name, image, detection_matrix, categories):
    voc_annotation_path = Path('voc/Annotations')
    complete_voc_annotation_path = Path(os.path.join(FLAGS.output, voc_annotation_path))
    complete_voc_annotation_path.mkdir(parents=True, exist_ok=True)

    voc_image_path = Path('voc/JPEGImages')
    complete_voc_image_path = Path(os.path.join(FLAGS.output, voc_image_path))
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

    for row in detection_matrix.T:
        object = ET.SubElement(root, 'object')

        name = ET.SubElement(object, 'name')
        name.text = categories[row[0]]['name']
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = str(0)

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(row[3]*im.width)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(row[2]*im.height)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(row[5]*im.width)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(row[4]*im.height)

    xml_string = ET.tostring(root, pretty_print=True)
    with open(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), 'wb') as files:
        files.write(xml_string)


def predict(argv):
    flags.mark_flag_as_required('export_dir')
    flags.mark_flag_as_required('output')
    flags.mark_flags_as_mutual_exclusive(flag_names=['image', 'image_folder', 'tfrecord'], required=True)

    saved_model_path = os.path.join(FLAGS.export_dir, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    categories = label_map_util.create_category_index_from_labelmap(FLAGS.label_map_path, use_display_name=True)


    # if FLAGS.image_folder is not None:
    #     for image_name in os.listdir(FLAGS.image_folder):
    #         image_path = os.path.join(FLAGS.image_folder, image_name)
    #         visualize_image(model=model, image_path=image_path, categories=categories)
    # elif FLAGS.image is not None:
    #     image_path = FLAGS.image
    #     visualize_image(model=model, image_path=image_path, categories=categories)
    if FLAGS.tfrecord is not None:
        records = read_tfrecord(FLAGS.tfrecord)
        for image, filename in zip(records['image'], records['filename']):
            visualize_image(model=model, image=image, image_name=filename, categories=categories)



if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    app.run(predict)
