# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
from absl import logging
import os
import json
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from fruitod.settings_gpu_0 import *
from sklearn.preprocessing import RobustScaler
import pickle


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid PNG
  """
    img_path = os.path.join(image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_pic = fid.read()
    # encoded_pic_io = io.BytesIO(encoded_pic)
    # image = PIL.Image.open(encoded_pic_io)
    # if image.format != 'PNG':
    #     raise ValueError('Image format not PNG')
    key = hashlib.sha256(encoded_pic).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    if 'weightInGrams' in data.keys():
        weight_in_grams = data['weightInGrams']
    else:
        weight_in_grams = -1.0

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/weightInGrams': dataset_util.float_feature(weight_in_grams),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_pic),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
    }))
    return example


def normalize_weights(weights_dir):
    weight_dict = {}
    for weight_file in os.listdir(weights_dir):
        print(weight_file)
        weight_file_complete = os.path.join(weights_dir, weight_file)
        with open(weight_file_complete) as f:
            json_dict = json.load(f)
        if 'weightInGrams' in json_dict.keys():
            weight_dict[weight_file] = json_dict['weightInGrams']

    weight_arr = np.array([val for val in weight_dict.values()])
    weight_max = np.amax(weight_arr)
    weight_min = np.amin(weight_arr)
    weight_mean = np.mean(weight_arr)
    weight_dict_normalized = {key: (weight_dict[key] - weight_min)/(weight_max - weight_min) for key in weight_dict.keys()}

    return weight_dict_normalized, weight_mean


def create_tfrecord(output_path,
                    data_path,
                    set,
                    add_weight_information=False,
                    ignore_difficult_instances=False):

    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])
    writer = tf.python_io.TFRecordWriter(output_path)

    complete_label_map_path = os.path.join(data_path, 'pascal_label_map.pbtxt')
    label_map_dict = label_map_util.get_label_map_dict(complete_label_map_path)

    examples_path = os.path.join(data_path, 'ImageSets', 'Main', set + '.txt')
    annotations_dir = os.path.join(data_path, 'Annotations')
    if add_weight_information:
        weights_dir = os.path.join(data_path, 'Weights')
        weights_dict = {}
        for weight_file in os.listdir(weights_dir):
            weight_file_without_extension = Path(weight_file).stem
            weight_file_complete = os.path.join(weights_dir, weight_file)
            with open(weight_file_complete) as f:
                json_dict = json.load(f)
            weights_dict[weight_file_without_extension] = json_dict['weightInGrams']

        scaler = RobustScaler()
        scaler.fit(list(weights_dict.values()))
        pickle.dump(scaler, open(os.path.join(data_path, 'scaler.pkl'), 'wb'))
        transformed_weights = scaler.transform(list(weights_dict.values()))

        normalized_weights_dict = dict(zip(list(weights_dict.keys()), list(transformed_weights)))

    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        example_without_extension = os.path.splitext(example)[0]
        if idx % 5 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example_without_extension + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        # read weight informations from voc/Weights/<file>.json
        if add_weight_information:
            data['weightInGrams'] = normalized_weights_dict[example_without_extension]

        tf_example = dict_to_tf_example(data, data_path, label_map_dict,
                                        ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main():
    create_tfrecord(output_path=TRAIN_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    add_weight_information=ADD_WEIGHT_INFORMATION,
                    set='train')

    create_tfrecord(output_path=VAL_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    add_weight_information=ADD_WEIGHT_INFORMATION,
                    set='val')


if __name__ == '__main__':
    main()
