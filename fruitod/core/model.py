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
        """
            Constructor.
        Args:
            checkpoint_path: Path where to save checkpoints while training.
            config_path: Path of config file of model
            export_path: Path to save exported model
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.export_path = export_path

    def train(self, steps_per_epoch, checkpoint_every_n_epochs=10):
        """
            Trains the model.
        Args:
            steps_per_epoch: Number of steps that are to be trained for one epoch
            checkpoint_every_n_epochs: Epoch interval in which to save a checkpoint while training
        """
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
        """
            Evaluates all Training Checkpoints.
        """
        model_lib_v2.eval_continuously(
            pipeline_config_path=self.config_path,
            model_dir=self.checkpoint_path,
            checkpoint_dir=self.checkpoint_path,
            postprocess_on_cpu=True
        )

    def export(self, add_weight_as_input=False):
        """
            Exports trained model with or without additional weight input
        Args:
            add_weight_as_input: If true, add weight 'gesamtgewicht' as additional input to input signature
                of exported model
        """
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        exporter_lib_v2.export_inference_graph(input_type='image_tensor',
                                               pipeline_config=pipeline_config,
                                               trained_checkpoint_dir=self.checkpoint_path,
                                               output_directory=self.export_path,
                                               use_side_inputs=add_weight_as_input,
                                               side_input_shapes='1',
                                               side_input_types='tf.float32',
                                               side_input_names='weightScaled')
