from fruitod.core.pipeline import Pipeline
from fruitod.core.create_tfrecord_from_voc import create_tfrecord
from fruitod.utils.file_util import get_steps_per_epoch
from fruitod.utils.csv_util import write_metrics
from fruitod.settings_gpu_0 import *


if __name__ == '__main__':
    create_tfrecord(output_path=TRAIN_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    set='train')

    create_tfrecord(output_path=TEST_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    set='val')

    steps_per_epoch = get_steps_per_epoch(tfrecord_path=TRAIN_TFRECORD_PATH, batch_size=BATCH_SIZE)

    config = Pipeline(config_path=CONFIG_PATH,
                      model_type=MODEL_TYPE)
    config.set_labelmap(labelmap_path=LABELMAP_PATH)
    config.set_train_tfrecord(train_tfrecord_path=TRAIN_TFRECORD_PATH)
    config.set_val_tfrecord(val_tfrecord_path=TEST_TFRECORD_PATH)
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
