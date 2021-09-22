import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np
from absl import flags, app
from fruitod.utils import file_util
from object_detection.utils import visualization_utils
from lxml import etree


flags.DEFINE_string('model_path', None, 'Path to model (has checkpoints directory)')
flags.DEFINE_string('images_path', None, 'Path to folder with images for prediction')
flags.DEFINE_string('voc_path', None, 'Path to voc dataset folder with labelmap and tfrecord '
                                      'and optional weights and scaler file')
flags.DEFINE_string('weight_path', None, 'Path to directory with weight json files')
flags.DEFINE_string('labelmap_file', 'label_map.pbtxt', 'Name of labelmap File')
flags.DEFINE_string('scaler_file', 'minmax_scaler.pkl', 'Filename of scaler pickle')
flags.DEFINE_boolean('side_input', False, 'wether to include weights as side input into model')
flags.DEFINE_float('iou_threshold', 0.95, 'IoU threshold for extra class agnostic NMS')
flags.DEFINE_float('score_threshold', 0.5, 'score threshold for extra class agnostic NMS and printing')

flags.mark_flags_as_required(['model_path', 'images_path', 'voc_path'])
FLAGS = flags.FLAGS


def inference(images_path,
              model_path,
              voc_path,
              labelmap_file,
              weight_path=None,
              scaler_file=None,
              side_input=False,
              iou_threshold=0.95,
              score_threshold=0.5):

    output_path = Path(os.path.join(images_path, 'prediction'))
    output_path.mkdir(parents=True, exist_ok=True)

    categories, labelmap_dict, category_index = file_util.load_labelmap(voc_path, labelmap_file)

    for image in os.listdir(images_path):

        image_without_extension = Path(image).stem

        if side_input:
            scaler = pickle.load(open(os.path.join(voc_path, scaler_file), 'rb'))
            weightfile_path = os.path.join(weight_path, image_without_extension + '.json')
            with open(weightfile_path) as f:
                json_dict = json.load(f)
            weight = json_dict['weightInGrams']
            weight_scaled = scaler.transform(np.asarray(weight).reshape(-1, 1))
            weight_scaled = np.asarray([np.squeeze(weight_scaled)], dtype=np.float32)

        image_path = os.path.join(images_path, image)
        image_np = _load_image_into_numpy_array(image_path)
        input_tensor = np.expand_dims(image_np, 0)

        if side_input:
            detections = model(input_tensor, weight_scaled)
        else:
            detections = model(input_tensor)

        detection_scores = detections['detection_scores'][0].numpy()
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)


        # Additional class agnostic NMS processing to select best box of object for multi label classification
        indices = tf.image.non_max_suppression(boxes=detection_boxes,
                                               scores=detection_scores,
                                               iou_threshold=iou_threshold,
                                               score_threshold=score_threshold,
                                               max_output_size=64)
        indices_np = indices.numpy()

        nmsed_detection_scores = detection_scores[indices_np]
        nmsed_detection_boxes = detection_boxes[indices_np]
        nmsed_detection_classes = detection_classes[indices_np]

        nmsed_detection_weights = None
        if 'detection_weightPerObject' in detections:
            detection_weights = detections['detection_weightPerObject'][0].numpy()
            nmsed_detection_weights = detection_weights[indices_np]

        image_np_with_detections = image_np.copy()
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes=nmsed_detection_boxes,
            classes=nmsed_detection_classes,
            scores=nmsed_detection_scores,
            weights=nmsed_detection_weights,
            category_index=category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=64,
            min_score_thresh=score_threshold,
            line_thickness=2
        )
        im = Image.fromarray(image_np_with_detections)
        image_save_path = Path(os.path.join(output_path, 'images_with_boxes'))
        image_save_path.mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(image_save_path, image))

        create_voc(output_path=output_path,
                   image=image_np,
                   image_name=image,
                   categories=categories,
                   detection_boxes=nmsed_detection_boxes,
                   detection_classes=nmsed_detection_classes,
                   detection_scores=nmsed_detection_scores)


def create_voc(output_path,
               image,
               image_name,
               categories,
               detection_boxes,
               detection_classes,
               detection_scores):
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

    for row, category, detection_score in zip(detection_boxes, detection_classes, detection_scores):
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
        xmin.text = str(row[1] * im.width)
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = str(row[0] * im.height)
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = str(row[3] * im.width)
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = str(row[2] * im.height)

    # xml_string = etree.tostring(root, pretty_print=True)
    # with open(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), 'wb') as files:
    #     files.write(xml_string)
    tree = etree.ElementTree(root)
    tree.write(os.path.join(complete_voc_annotation_path, image_name_without_extension + '.xml'), pretty_print=True)




def main(_):
    model_path = FLAGS.model_path
    images_path = FLAGS.images_path
    voc_path = FLAGS.voc_path
    weight_path = FLAGS.weight_path
    labelmap_file = FLAGS.labelmap_file
    scaler_file = FLAGS.scaler_file
    side_input = FLAGS.side_input
    iou_threshold = FLAGS.iou_threshold
    score_threshold = FLAGS.score_threshold

    inference(images_path=images_path,
              model_path=model_path,
              voc_path=voc_path,
              labelmap_file=labelmap_file,
              weight_path=weight_path,
              scaler_file=scaler_file,
              side_input=side_input,
              iou_threshold=iou_threshold,
              score_threshold=score_threshold)



if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    app.run(main)
