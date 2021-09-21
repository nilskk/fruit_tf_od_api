import os
from pathlib import Path

# Variables to customize
VOC_PATH = os.path.join(Path.home(), 'rewe_project/data/voc_weight_test')
PIPELINE_PATH = os.path.join(Path.home(), 'rewe_project/models/models_server/prob_two_stage/efficientdet_d0_iou0.5_score0.7')
# Just an example. Choose best model directory name on own preferences to ensure uniqueness
MODEL_DIR = 'model_lr=0.001_input-multiply'

# !!!!! IMPORTANT !!!!!: set right MODEL_TYPE depending on pipeline.config('ssd' or 'prob_two_stage').
# Otherwise it could falsely override pipeline.config
MODEL_TYPE = "prob_two_stage"

ADD_WEIGHT_AS_INPUT = False     # Set to True if you want to add the whole weight as input

# One of ['input-multiply', 'fpn-multiply', 'roi-multiply', 'concat']
# For 'ssd' only 'input-multiply' and 'fpn-multiply' is supported
INPUT_METHOD = 'input-multiply'

SCALER_METHOD = 'minmax'        # Name of weight scaling method

ADD_WEIGHT_AS_OUTPUT_GPO = False        # Set to True if you want to add weight per object as additional output
ADD_WEIGHT_AS_OUTPUT_GESAMT = False     # (Only works for 'ssd') Set to True if you want to add whole weight as output

NUM_CLASSES = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 30
FIRST_DECAY_EPOCHS = 10
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'decay'


# Inferred Variables
TFRECORDS_PATH = VOC_PATH
TRAIN_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'train_gpu_0.tfrecord')
TEST_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'test_gpu_0.tfrecord')
LABELMAP_PATH = os.path.join(VOC_PATH, 'label_map.pbtxt')

CONFIG_PATH = os.path.join(PIPELINE_PATH, 'pipeline.config')

MODEL_NAME = os.path.basename(PIPELINE_PATH)

MODEL_PATH = os.path.join(PIPELINE_PATH, MODEL_DIR)
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')
EXPORT_PATH = os.path.join(MODEL_PATH, 'export')
PREDICTION_OUTPUT_PATH = os.path.join(MODEL_PATH, 'prediction')
