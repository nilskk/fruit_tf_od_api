from fruitod.core.model import Model
from fruitod.core.pipeline import Pipeline
from fruitod.core.create_tfrecord_from_voc import create_tfrecord
from fruitod.utils.file_util import get_steps_per_epoch
from fruitod.utils.csv_util import write_metrics
from absl import logging
import tensorflow as tf
from argparse import ArgumentParser
import datetime
import time


if __name__ == '__main__':
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)

    # tf.config.run_functions_eagerly(True)
    # tf.config.set_soft_device_placement(True)

    if args.gpu == 0:
        from fruitod.settings_gpu_0 import *
    elif args.gpu == 1:
        from fruitod.settings_gpu_1 import *

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    create_tfrecord(output_path=TRAIN_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    add_weight_information=ADD_WEIGHT_INFORMATION,
                    set='train')

    create_tfrecord(output_path=VAL_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    add_weight_information=ADD_WEIGHT_INFORMATION,
                    set='val')

    steps_per_epoch = get_steps_per_epoch(tfrecord_path=TRAIN_TFRECORD_PATH, batch_size=BATCH_SIZE)

    config = Pipeline(config_path=CONFIG_PATH,
                      model_type=MODEL_TYPE)
    config.set_labelmap(labelmap_path=LABELMAP_PATH)
    config.set_train_tfrecord(train_tfrecord_path=TRAIN_TFRECORD_PATH)
    config.set_val_tfrecord(val_tfrecord_path=VAL_TFRECORD_PATH)
    config.set_model_name(model_name=MODEL_NAME)
    config.set_optimizer(optimizer_name=OPTIMIZER_NAME,
                         first_decay_epochs=FIRST_DECAY_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         learning_rate=LEARNING_RATE)
    config.set_batch_size(batch_size=BATCH_SIZE)
    config.set_num_classes(num_classes=NUM_CLASSES)
    config.set_train_epochs(train_epochs=TRAIN_EPOCHS,
                            steps_per_epoch=steps_per_epoch)

    metrics = {'Name': MODEL_NAME,
               'Optimizer': OPTIMIZER_NAME,
               'Batch Size': BATCH_SIZE,
               'Learning Rate': LEARNING_RATE}
    write_metrics(SAVE_PATH, metrics)

    model = Model(checkpoint_path=CHECKPOINT_PATH,
                  config_path=CONFIG_PATH,
                  export_path=EXPORT_PATH)

    model.train(batch_size=BATCH_SIZE, checkpoint_every_n_epochs=10)

    model.evaluate()
    
    model.export(add_weight_information=ADD_WEIGHT_INFORMATION)

    end = time.time()
    time_diff = end - start
    print("Laufzeit: " + str(datetime.timedelta(seconds=time_diff)))
