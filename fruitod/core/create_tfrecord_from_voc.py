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

r"""Convert VOTT dataset to TFRecord for object_detection.

Example usage:
    python create_tfrecord_from_voc.py \
        --data_dir=pascal_data \
        --set=train \
        --output_dir=tfrecords \
        --vott_sourceconnection_name=Mango \
        """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# flags = tf.app.flags
# flags.DEFINE_string('data_path', '', 'Root directory to raw PASCAL VOC dataset.')
# flags.DEFINE_string('set', 'train', 'Convert training set or validation set.')
# flags.DEFINE_string('voc_set_name', 'vott', 'Name of the set in voc for naming of '
#                                                           'train and val txt files.')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
#                                                           'difficult instances')
# FLAGS = flags.FLAGS
# 
# SETS = ['train', 'val']


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

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
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
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
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
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tfrecord(output_path,
                    data_path,
                    set,
                    voc_set_name,
                    ignore_difficult_instances=False):

    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])
    writer = tf.python_io.TFRecordWriter(output_path)

    complete_label_map_path = os.path.join(data_path, 'pascal_label_map.pbtxt')
    label_map_dict = label_map_util.get_label_map_dict(complete_label_map_path)

    examples_path = os.path.join(data_path, 'ImageSets', 'Main',
                                 voc_set_name + "_" + set + '.txt')
    annotations_dir = os.path.join(data_path, 'Annotations')
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

        tf_example = dict_to_tf_example(data, data_path, label_map_dict,
                                        ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    # if FLAGS.set not in SETS:
    #     raise ValueError('set must be in : {}'.format(SETS))

    create_tfrecord(output_path=FLAGS.output_path,
                    data_path=FLAGS.data_path,
                    set=FLAGS.set,
                    voc_set_name=FLAGS.voc_set_name,
                    ignore_difficult_instances=FLAGS.ignore_difficult_instances)


if __name__ == '__main__':
    # tf.app.run()
    main()