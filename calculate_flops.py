import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import numpy as np


MODEL_PATH = "./models/exported_models/mobilenetv2_5classes_lr=0.01_bs=16_snms=0.5_iou=0.6"
SAVED_MODEL_PATH = MODEL_PATH + "/saved_model"


if __name__=="__main__":

    model = tf.saved_model.load(SAVED_MODEL_PATH)

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        print("Flops: {}".format(flops.total_float_ops))
