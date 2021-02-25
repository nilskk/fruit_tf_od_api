from fruitod.core.pipeline import Pipeline
from fruitod.core.create_tfrecord_from_voc import create_tfrecord
from fruitod.utils.file_util import get_steps_per_epoch
from settings import *


if __name__ == '__main__':
    create_tfrecord(output_path=TRAIN_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    set='train',
                    voc_set_name=VOC_SET_NAME)

    create_tfrecord(output_path=VAL_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    set='val',
                    voc_set_name=VOC_SET_NAME)

    steps_per_epoch = get_steps_per_epoch(tfrecord_path=TRAIN_TFRECORD_PATH, batch_size=BATCH_SIZE)

    config = Pipeline(config_path=CONFIG_PATH)
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
    config.set_train_epochs(train_epochs=TRAIN_EPOCHS)
