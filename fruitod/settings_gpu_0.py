import os
from pathlib import Path

# Variables to customize
HOME = os.path.join(Path.home(), 'rewe_project')
VOC_PATH = os.path.join(HOME, 'data/voc_weight_test')
MODEL_PATH = os.path.join(HOME, 'models/models_server/ssd/efficientdet_d0_test')

SCALER_METHOD = 'minmax'
ADD_WEIGHT_INFORMATION = False
WEIGHT_METHOD = 'input-multiply'
ADD_WEIGHT_AS_OUTPUT = False
ADD_WEIGHT_AS_OUTPUT_V2 = False

MODEL_TYPE = "ssd"
NUM_CLASSES = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.01
TRAIN_EPOCHS = 10
FIRST_DECAY_EPOCHS = 10
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'decay'

SCORE_THRESHOLD = 0.90
IOU_THRESHOLD = 0.5
VISUALIZE = True



# Inferred Variables
TFRECORDS_PATH = VOC_PATH
TRAIN_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'train_gpu_0.tfrecord')
VAL_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'val_gpu_0.tfrecord')
LABELMAP_PATH = os.path.join(VOC_PATH, 'pascal_label_map.pbtxt')

CONFIG_PATH = os.path.join(MODEL_PATH, 'pipeline.config')

MODEL_NAME = os.path.basename(MODEL_PATH)

SAVE_PATH = os.path.join(MODEL_PATH,
                         'lr={}_bs={}_classes={}_{}'.format(LEARNING_RATE,
                                                                BATCH_SIZE,
                                                                NUM_CLASSES, OPTIMIZER_NAME))
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints')
EXPORT_PATH = os.path.join(SAVE_PATH, 'export')
PREDICTION_OUTPUT_PATH = os.path.join(SAVE_PATH, 'prediction')

