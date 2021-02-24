import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
import time
from fruitod.utils.csv_util import write_metrics
from fruitod.utils.file_util import read_tfrecord
from absl import flags
import numpy as np


# flags.DEFINE_string('export_path', None, 'Path to exported model')
# flags.DEFINE_string('tfrecord_path', './data/tfrecords/vott_val.tfrecord',
#                     'Path to tfrecord')
# 
# FLAGS = flags.FLAGS


def inference_on_single_image(model, image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]
    tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

    start = time.time()
    detections = model(image_tensor)
    end = time.time()
    time_diff = end - start
    return time_diff


def inference(export_path,
              tfrecord_path):
    tf.config.set_visible_devices([], 'GPU')

    saved_model_path = os.path.join(export_path, 'saved_model')
    model = tf.saved_model.load(saved_model_path)

    time_diffs = []
    records = read_tfrecord(tfrecord_path)

    for record in records:
        time_diff = inference_on_single_image(model=model, image=record['image'])
        print(time_diff)
        time_diffs.append(time_diff)

    head, tail = os.path.split(str(export_path))

    time_diffs_array = np.array(time_diffs)

    average_inference_speed = np.average(time_diffs_array[1:])
    print("Average Inference Speed: " + str(average_inference_speed))

    metrics = {'Inference Speed': average_inference_speed}
    write_metrics(head, metrics)


def flops(export_path):
    tf.config.set_visible_devices([], 'GPU')

    saved_model_path = os.path.join(export_path, 'saved_model')
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
      
        head, tail = os.path.split(str(export_path))
        metrics = {'Flops': flops.total_float_ops}
        write_metrics(head, metrics)


def main(argv):
    # flags.mark_flag_as_required('export_path')

    flops(export_path=FLAGS.export_path)
    inference(export_path=FLAGS.export_path,
              tfrecord_path=FLAGS.tfrecord_path)


if __name__ == "__main__":
    # tf.compat.v1.app.run(main)
    main()