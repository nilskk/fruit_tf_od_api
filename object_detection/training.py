from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                                       'where event and checkpoint files will be written.')
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                                            'writing resulting metrics to `model_dir`.')

FLAGS = flags.FLAGS


def train(argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tf.config.set_soft_device_placement(True)

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

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.model_dir)


if __name__ == "__main__":
    tf.compat.v1.app.run(train)
