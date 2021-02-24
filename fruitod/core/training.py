import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
import math
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from fruitod.settings import *

# flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
# flags.DEFINE_string('model_path', None, 'Path to output model directory '
#                                        'where event and checkpoint files will be written.')
# flags.DEFINE_integer('checkpoint_every_n_epochs', 10, 'Number of epochs until next checkpoint')
# flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
#
# FLAGS = flags.FLAGS


def train(checkpoint_path,
          config_path,
          checkpoint_every_n_epochs=10,
          batch_size=8):

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
            model_dir=checkpoint_path,
            checkpoint_every_n=checkpoints_every_n_steps,
            checkpoint_max_to_keep=150)


def main(argv):
    # flags.mark_flag_as_required('model_path')
    # flags.mark_flag_as_required('config_path')

    train(checkpoint_path=CHECKPOINT_PATH,
          config_path=CONFIG_PATH,
          checkpoint_every_n_epochs=10,
          batch_size=BATCH_SIZE)


if __name__ == "__main__":
    # tf.compat.v1.app.run(main)
    main()
