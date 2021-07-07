import os
import math
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection import model_lib_v2
from object_detection import exporter_lib_v2
import tensorflow as tf
from absl import logging


class Model:
    def __init__(self, checkpoint_path, config_path, export_path):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.export_path = export_path

    def train(self, checkpoint_every_n_epochs=10, batch_size=8):
        pipeline = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.config_path, 'r') as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline)

        train_tfrecords_path = pipeline.train_input_reader.tf_record_input_reader.input_path
        num_train_images = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecords_path))
        steps_per_epoch = math.ceil(num_train_images / batch_size)
        checkpoints_every_n_steps = steps_per_epoch * checkpoint_every_n_epochs

        strategy = tf.compat.v2.distribute.MirroredStrategy()
        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=self.config_path,
                model_dir=self.checkpoint_path,
                checkpoint_every_n=checkpoints_every_n_steps,
                checkpoint_max_to_keep=150,
                record_summaries=True)

    def evaluate(self):
        model_lib_v2.eval_continuously(
            pipeline_config_path=self.config_path,
            model_dir=self.checkpoint_path,
            checkpoint_dir=self.checkpoint_path,
            postprocess_on_cpu=True
        )

    def export(self, add_weight_information=False):
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        exporter_lib_v2.export_inference_graph(input_type='image_tensor',
                                               pipeline_config=pipeline_config,
                                               trained_checkpoint_dir=self.checkpoint_path,
                                               output_directory=self.export_path,
                                               use_side_inputs=add_weight_information,
                                               side_input_shapes='1',
                                               side_input_types='tf.float32',
                                               side_input_names='weightInGrams')
