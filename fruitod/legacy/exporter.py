import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from absl import app
from absl import flags


# flags.DEFINE_string('config_path', None,
#                     'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                     'file.')
# flags.DEFINE_string('checkpoint_path', None,
#                     'Path to trained checkpoint directory')
# flags.DEFINE_string('export_path', None, 'Path to write outputs.')
#
# FLAGS = flags.FLAGS


def export(config_path,
           checkpoint_path,
           export_path):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    exporter_lib_v2.export_inference_graph(input_type='image_tensor',
                                           pipeline_config=pipeline_config,
                                           trained_checkpoint_dir=checkpoint_path,
                                           output_directory=export_path)


def main(argv):
    # flags.mark_flag_as_required('config_path')
    # flags.mark_flag_as_required('checkpoint_path')
    # flags.mark_flag_as_required('export_path')

    export(config_path=FLAGS.config_path,
           checkpoint_path=FLAGS.checkpoint_path,
           export_path=FLAGS.export_path)


if __name__ == '__main__':
    # app.run(main)
    main()
