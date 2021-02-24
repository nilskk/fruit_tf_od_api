import os
from fruitod.core.create_tfrecord_from_voc import create_tfrecord
from fruitod.core.training import train
from fruitod.core.evaluation import evaluate
from fruitod.core.change_config import change_pipeline
from fruitod.core.collect_summary import collect_csv
from fruitod.core.calculate_performance import flops, inference
from fruitod.core.predict_image import predict
from fruitod.core.exporter import export
from fruitod.settings import *
import tensorflow as tf


if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
                    set='train',
                    voc_set_name='vott')

    create_tfrecord(output_path=VAL_TFRECORD_PATH,
                    data_path=VOC_PATH,
                    set='val',
                    voc_set_name='vott')

    change_pipeline(checkpoint_path=CHECKPOINT_PATH,
                    config_path=CONFIG_PATH,
                    labelmap_path=LABELMAP_PATH,
                    train_tfrecord_path=TRAIN_TFRECORD_PATH,
                    val_tfrecord_path=VAL_TFRECORD_PATH,
                    model_name=MODEL_NAME,
                    optimizer_name=OPTIMIZER_NAME,
                    num_classes=NUM_CLASSES,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                    train_epochs=TRAIN_EPOCHS,
                    first_decay_epochs=FIRST_DECAY_EPOCHS)
    
    train(checkpoint_path=CHECKPOINT_PATH,
          config_path=CONFIG_PATH,
          batch_size=BATCH_SIZE,
          checkpoint_every_n_epochs=10)

    evaluate(checkpoint_path=CHECKPOINT_PATH,
             config_path=CONFIG_PATH)
    
    export(config_path=CONFIG_PATH,
           checkpoint_path=CHECKPOINT_PATH,
           export_path=EXPORT_PATH)
    
    inference(export_path=EXPORT_PATH,
              tfrecord_path=VAL_TFRECORD_PATH)
    
    flops(export_path=EXPORT_PATH)
    
    collect_csv(models_path=os.path.split(MODEL_PATH)[0])

    predict(export_path=EXPORT_PATH,
            output_path=PREDICTION_OUTPUT_PATH,
            labelmap_path=LABELMAP_PATH,
            tfrecord_path=VAL_TFRECORD_PATH,
            score_threshold=SCORE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            visualize=VISUALIZE)

