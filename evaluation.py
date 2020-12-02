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


def evaluate(argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    flags.mark_flag_as_required('checkpoint_dir')

    tf.config.set_visible_devices([], 'GPU')
    print(tf.config.get_visible_devices())

    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        checkpoint_dir=FLAGS.checkpoint_dir,
        postprocess_on_cpu=True,
        wait_interval=1,
        timeout=400)


if __name__ == "__main__":
    tf.compat.v1.app.run(evaluate)
