import tensorflow as tf
import math
import os
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def load_model(model_path):
    """
        Load exported model for inference
    Args:
        model_path: Path of model directory

    Returns:
        model: SavedModel of exported model

    """
    saved_model_path = os.path.join(model_path, 'export', 'saved_model')
    model = tf.saved_model.load(saved_model_path)
    return model


def load_labelmap(voc_path, labelmap_file):
    """
        Loads labelmap file and reads data to different data structures
    Args:
        voc_path: Path to used VOC dataset
        labelmap_file: String containing the Name of labelmap_file

    Returns:
        categories: a list of dictionaries representing all possible categories
        labelmap_dict: a dictionary mapping label names to id
        category_index: a category index, which is a dictionary that maps integer ids to dicts containing categories

    """
    labelmap_path = os.path.join(voc_path, labelmap_file)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path)
    categories = label_map_util.create_categories_from_labelmap(labelmap_path)
    labelmap_dict = label_map_util.get_label_map_dict(labelmap_path)

    return categories, labelmap_dict, category_index


def get_steps_per_epoch(train_tfrecord_path,
                        batch_size):
    """
        Calculate steps per epochs
    Args:
        train_tfrecord_path: Path of training tfrecord
        batch_size: an Integer for the batch size used for training

    Returns:
        steps_per_epoch: Number of steps for one epoch

    """
    num_train_images = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord_path))
    steps_per_epoch = math.ceil(num_train_images / batch_size)

    return steps_per_epoch


def read_config(config_path):
    """
        Read pipeline.config into data structure
    Args:
        config_path: Path to config file

    Returns:
        pipeline: protobuf data structure of config

    """
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)

    return pipeline


def write_config(pipeline,
                 config_path):
    """
        Writes protobuf pipeline to pipeline.config file
    Args:
        pipeline: protobuf data structure of config
        config_path: Path to config file
    """
    pipeline_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(config_path, 'wb') as f:
        f.write(pipeline_text)
