from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

# flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
# flags.DEFINE_string('model_path', None, 'Path to output model directory '
#                                        'where event and checkpoint files will be written.')
# flags.DEFINE_string('checkpoint_path', None, 'Path to directory holding a checkpoint.  If '
#                                             '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
#                                             'writing resulting metrics to `model_dir`.')
#
# FLAGS = flags.FLAGS


def evaluate(checkpoint_path,
             config_path):
    tf.config.set_visible_devices([], 'GPU')

    model_lib_v2.eval_continuously(
        pipeline_config_path=config_path,
        model_dir=checkpoint_path,
        checkpoint_dir=checkpoint_path,
        postprocess_on_cpu=True
    )


def main(argv):
    # flags.mark_flag_as_required('model_path')
    # flags.mark_flag_as_required('config_path')
    # flags.mark_flag_as_required('checkpoint_path')

    evaluate(config_path=FLAGS.config_path, checkpoint_path=FLAGS.checkpoint_path)


if __name__ == "__main__":
    # tf.compat.v1.app.run(main)
    main()
