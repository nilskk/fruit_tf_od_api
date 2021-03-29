import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from absl import flags
from fruitod.utils.csv_util import write_metrics
from fruitod.utils.file_util import get_steps_per_epoch
import os
from pathlib import Path

# flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
# flags.DEFINE_string('model_path', None, 'Path to model directory')
# flags.DEFINE_string('model_name', "", 'Name of the model')
# flags.DEFINE_integer('train_epochs', 150, 'Number of epochs to train')
# flags.DEFINE_integer('first_decay_epochs', 10, 'Number of epochs for first cosine decay')
# flags.DEFINE_integer('num_classes', 5, 'Number of classes in model')
# flags.DEFINE_integer('batch_size', 16, 'Batch size for training')
# flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use')
# flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate')
# flags.DEFINE_string('labelmap_path', "./data/voc_data/pascal_label_map.pbtxt", 'Path to pascal_label_map.pbtxt')
# flags.DEFINE_string('train_tfrecord_path', "./data/tfrecords/vott_train.tfrecord", 'Path to train tfrecord file')
# flags.DEFINE_string('val_tfrecord_path', "./data/tfrecords/vott_val.tfrecord", 'Path to val tfrecord file')
# 
# FLAGS = flags.FLAGS


def read_config(config_path):
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)

    return pipeline


def write_config(pipeline,
                 config_path):
    pipeline_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(config_path, 'wb') as f:
        f.write(pipeline_text)


def set_model_name(pipeline,
                   model_name):
    pipeline.model.name = model_name


def set_num_classes(pipeline,
                    num_classes):
    pipeline.model.ssd.num_classes = num_classes


def set_batch_size(pipeline,
                   batch_size):
    pipeline.train_config.batch_size = batch_size


def set_train_epochs(pipeline,
                     train_epochs,
                     steps_per_epoch):
    pipeline.train_config.num_steps = train_epochs * steps_per_epoch


def set_optimizer(pipeline,
                  optimizer_name,
                  first_decay_epochs,
                  steps_per_epoch,
                  learning_rate):
    if optimizer_name == 'adam':
        optimizer = pipeline.train_config.optimizer.adam_optimizer
        pipeline.train_config.optimizer.adam_optimizer.epsilon = 1e-8
    elif optimizer_name == 'sgd':
        optimizer = pipeline.train_config.optimizer.momentum_optimizer
        pipeline.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

    optimizer.learning_rate.cosine_restart_learning_rate.first_decay_steps = first_decay_epochs * steps_per_epoch
    optimizer.learning_rate.cosine_restart_learning_rate.initial_learning_rate = learning_rate


def set_labelmap(pipeline,
                 labelmap_path):
    pipeline.train_input_reader.label_map_path = labelmap_path
    pipeline.eval_input_reader[0].label_map_path = labelmap_path


def set_train_tfrecord(pipeline,
                       train_tfrecord_path):
    pipeline.train_input_reader.tf_record_input_reader.input_path[0] = train_tfrecord_path


def set_val_tfrecord(pipeline,
                     val_tfrecord_path):
    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = val_tfrecord_path


def change_pipeline(checkpoint_path,
                    config_path,
                    labelmap_path,
                    train_tfrecord_path,
                    val_tfrecord_path,
                    model_name,
                    optimizer_name,
                    num_classes,
                    batch_size,
                    learning_rate,
                    train_epochs,
                    first_decay_epochs):

    pipeline = read_config(config_path=config_path)

    steps_per_epoch = get_steps_per_epoch(tfrecord_path=train_tfrecord_path, batch_size=batch_size)

    set_model_name(pipeline, model_name=model_name)
    set_num_classes(pipeline, num_classes=num_classes)
    set_batch_size(pipeline, batch_size=batch_size)
    set_train_epochs(pipeline, train_epochs=train_epochs, steps_per_epoch=steps_per_epoch)
    set_optimizer(pipeline,
                  optimizer_name=optimizer_name,
                  first_decay_epochs=first_decay_epochs,
                  steps_per_epoch=steps_per_epoch,
                  learning_rate=learning_rate)
    set_labelmap(pipeline, labelmap_path=labelmap_path)
    set_train_tfrecord(pipeline, train_tfrecord_path=train_tfrecord_path)
    set_val_tfrecord(pipeline, val_tfrecord_path=val_tfrecord_path)

    write_config(pipeline, config_path=config_path)

    head, tail = os.path.split(checkpoint_path)
    Path(head).mkdir(parents=True, exist_ok=True)

    metrics = {'Name': model_name,
               'Optimizer': optimizer_name,
               'Batch Size': batch_size,
               'Learning Rate': learning_rate}
    write_metrics(head, metrics)


def main(argv):
    # flags.mark_flag_as_required('config_path')
    # flags.mark_flag_as_required('model_path')

    change_pipeline(checkpoint_path=FLAGS.model_path,
                    config_path=FLAGS.config_path,
                    labelmap_path=FLAGS.labelmap_path,
                    train_tfrecord_path=FLAGS.train_tfrecord_path,
                    val_tfrecord_path=FLAGS.val_tfrecord_path,
                    model_name=FLAGS.model_name,
                    optimizer_name=FLAGS.optimizer,
                    num_classes=FLAGS.num_classes,
                    batch_size=FLAGS.batch_size,
                    learning_rate=FLAGS.learning_rate,
                    train_epochs=FLAGS.train_epochs,
                    first_decay_epochs=FLAGS.first_decay_epochs)


if __name__ == "__main__":
    # tf.compat.v1.app.run(main)
    main()
