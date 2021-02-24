from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
import math
import os
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_path', None, 'Path to output model directory '
                                       'where event and checkpoint files will be written.')
flags.DEFINE_integer('checkpoint_every_n_epochs', 10, 'Number of epochs until next checkpoint')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')

FLAGS = flags.FLAGS


def train(model_path,
          config_path,
          checkpoint_every_n_epochs=10,
          batch_size=8):

    tf.config.set_soft_device_placement(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)

    train_tfrecords_path = pipeline.train_input_reader.tf_record_input_reader.input_path
    num_train_images = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecords_path))
    steps_per_epoch = math.ceil(num_train_images / batch_size)
    checkpoints_every_n_steps = steps_per_epoch * checkpoint_every_n_epochs

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=config_path,
            model_dir=model_path,
            checkpoint_every_n=checkpoints_every_n_steps,
            checkpoint_max_to_keep=150)


def main(argv):
    flags.mark_flag_as_required('model_path')
    flags.mark_flag_as_required('config_path')

    train(model_path=FLAGS.model_path,
          config_path=FLAGS.config_path,
          checkpoint_every_n_epochs=FLAGS.checkpoint_every_n_epochs,
          batch_size=FLAGS.batch_size)


if __name__ == "__main__":
    tf.compat.v1.app.run(main)
