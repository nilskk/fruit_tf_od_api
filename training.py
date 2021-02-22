from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
import math
import os

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                                       'where event and checkpoint files will be written.')
flags.DEFINE_string('train_tfrecords_path', "./data/tfrecords/vott_train.tfrecord", 'Path to train tfrecord file')
flags.DEFINE_integer('checkpoint_every_n_epochs', 10, 'Number of epochs until next checkpoint')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training')

FLAGS = flags.FLAGS


def train(argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
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

    num_train_images = sum(1 for _ in tf.data.TFRecordDataset(FLAGS.train_tfrecords_path))
    steps_per_epoch = math.ceil(num_train_images / FLAGS.batch_size)
    checkpoints_every_n_steps = steps_per_epoch * FLAGS.checkpoint_every_n_epochs

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.model_dir,
            checkpoint_every_n=checkpoints_every_n_steps,
            checkpoint_max_to_keep=150)


if __name__ == "__main__":
    tf.compat.v1.app.run(train)
