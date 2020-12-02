import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from absl import flags
from csv_util import create_dataframe, write_bs, write_lr, write_name
import os

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to model directory')
flags.DEFINE_integer('train_steps', 25000, 'Number of steps to train')
flags.DEFINE_integer('cosine_steps', 25000, 'Number of steps for cosine decay')
flags.DEFINE_integer('warmup_steps', 1000, 'Number of steps for warmup')
flags.DEFINE_integer('num_classes', 14, 'Number of classes in model')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training')
flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate for cosine decay')
flags.DEFINE_string('label_map_path', "./data/voc_data/pascal_label_map.pbtxt", 'Path to pascal_label_map.pbtxt')
flags.DEFINE_string('train_tfrecords_path', "./data/tfrecords/vott_train.tfrecord", 'Path to train tfrecord file')
flags.DEFINE_string('val_tfrecords_path', "./data/tfrecords/vott_val.tfrecord", 'Path to val tfrecord file')

FLAGS = flags.FLAGS


def change_pipeline(argv):
    flags.mark_flag_as_required('pipeline_config_path')
    flags.mark_flag_as_required('model_dir')
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)

    pipeline.model.ssd.num_classes = FLAGS.num_classes

    pipeline.train_config.num_steps = FLAGS.train_steps
    pipeline.train_config.batch_size = FLAGS.batch_size
    pipeline.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = FLAGS.cosine_steps
    pipeline.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = FLAGS.warmup_steps
    pipeline.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = FLAGS.learning_rate

    pipeline.train_input_reader.tf_record_input_reader.input_path[0] = FLAGS.train_tfrecords_path
    pipeline.train_input_reader.label_map_path = FLAGS.label_map_path

    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = FLAGS.val_tfrecords_path
    pipeline.eval_input_reader[0].label_map_path = FLAGS.label_map_path

    pipeline_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'wb') as f:
        f.write(pipeline_text)
    
    head, tail = os.path.split(FLAGS.model_dir)

    name = pipeline.model.ssd.feature_extractor.type
    write_name(head, name)
    write_bs(head, FLAGS.batch_size)
    write_lr(head, FLAGS.learning_rate)

if __name__=="__main__":
    tf.compat.v1.app.run(change_pipeline)

