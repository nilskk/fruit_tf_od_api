import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from absl import flags
from csv_util import create_dataframe, write_bs, write_lr, write_name, write_optimizer
import os
import math

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to model directory')
flags.DEFINE_string('model_name', "", 'Name of the model')
flags.DEFINE_integer('train_epochs', 150, 'Number of epochs to train')
flags.DEFINE_integer('first_decay_epochs', 10, 'Number of epochs for first cosine decay')
flags.DEFINE_integer('num_classes', 5, 'Number of classes in model')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training')
flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use')
flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate')
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

    num_train_images = sum(1 for _ in tf.data.TFRecordDataset(FLAGS.train_tfrecords_path))
    steps_per_epoch = math.ceil(num_train_images / FLAGS.batch_size)

    pipeline.model.name = FLAGS.model_name

    pipeline.model.ssd.num_classes = FLAGS.num_classes

    pipeline.train_config.num_steps = FLAGS.train_epochs * steps_per_epoch
    pipeline.train_config.batch_size = FLAGS.batch_size

    if FLAGS.optimizer == 'adam':
        optimizer = pipeline.train_config.optimizer.adam_optimizer
        pipeline.train_config.optimizer.adam_optimizer.epsilon = 1e-8
    elif FLAGS.optimizer == 'sgd':
        optimizer = pipeline.train_config.optimizer.momentum_optimizer
        pipeline.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

    optimizer.learning_rate.cosine_restart_learning_rate.first_decay_steps = FLAGS.first_decay_epochs * steps_per_epoch
    optimizer.learning_rate.cosine_restart_learning_rate.initial_learning_rate = FLAGS.learning_rate

    pipeline.train_input_reader.tf_record_input_reader.input_path[0] = FLAGS.train_tfrecords_path
    pipeline.train_input_reader.label_map_path = FLAGS.label_map_path

    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = FLAGS.val_tfrecords_path
    pipeline.eval_input_reader[0].label_map_path = FLAGS.label_map_path

    pipeline_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'wb') as f:
        f.write(pipeline_text)
    
    head, tail = os.path.split(FLAGS.model_dir)
    optimizer_name = FLAGS.optimizer
    create_dataframe(head)
    write_name(head, FLAGS.model_name)
    write_optimizer(head, optimizer_name)
    write_bs(head, FLAGS.batch_size)
    write_lr(head, FLAGS.learning_rate)

if __name__=="__main__":
    tf.compat.v1.app.run(change_pipeline)

