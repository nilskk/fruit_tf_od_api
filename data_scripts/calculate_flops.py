import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
from util.csv_util import write_flops
from absl import flags


flags.DEFINE_string('export_dir', None, 'Path to exported model')

FLAGS = flags.FLAGS


def flops(argv):
    flags.mark_flag_as_required('export_dir')
    saved_model_path = os.path.join(FLAGS.export_dir, 'saved_model')
    
    model = tf.saved_model.load(saved_model_path)

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        print("Flops: {}".format(flops.total_float_ops))
      
        head, tail = os.path.split(str(FLAGS.export_dir))
        write_flops(head, flops.total_float_ops)
    

if __name__ == "__main__":
    tf.compat.v1.app.run(flops)
