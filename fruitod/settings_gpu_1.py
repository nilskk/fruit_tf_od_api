import os
from pathlib import Path
##########################
# Variables to customize #
##########################

# Path to VOC dataset directory
VOC_PATH = os.path.join('/data', '<VOC_DIR>')

# Path to directory with pipeline.config
PIPELINE_PATH = os.path.join('/data', 'models/<PIPELINE_DIR>')

# Just an example. Choose best model directory name on own preferences to ensure uniqueness
MODEL_DIR = 'model_lr=0.001_input-multiply'
MODEL_NAME = '<Modelname>'  # Change to unique name for identifying metrics later

# !!!!! IMPORTANT !!!!!: set right MODEL_TYPE depending on pipeline.config; ('ssd' or 'prob_two_stage').
# Otherwise it could falsely override pipeline.config
MODEL_TYPE = 'prob_two_stage'

ADD_WEIGHT_AS_INPUT = False     # Set to True if you want to add the whole weight as input

# One of ['input-multiply', 'fpn-multiply', 'roi-multiply', 'concat']
# For 'ssd' only 'input-multiply' and 'fpn-multiply' is supported
INPUT_METHOD = 'input-multiply'

SCALER_METHOD = 'minmax'        # Name of weight scaling method

ADD_WEIGHT_AS_OUTPUT_GPO = False        # Set to True if you want to add weight per object as additional output
ADD_WEIGHT_AS_OUTPUT_GESAMT = False     # (Only works for 'ssd') Set to True if you want to add whole weight as output

NUM_CLASSES = 5             # Number of classes (without background class!)
BATCH_SIZE = 8              # Batch size for training
LEARNING_RATE = 0.001       # Base Learning Rate
TRAIN_EPOCHS = 30           # Number of Total Epochs for Training
FIRST_DECAY_EPOCHS = 10     # Number of Warmup Epochs for CosineDecay or CosineDecayWithRestarts
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'decay'


# Inferred Variables
TFRECORDS_PATH = VOC_PATH
TRAIN_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'train_gpu_1.tfrecord')
TEST_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'test_gpu_1.tfrecord')
LABELMAP_PATH = os.path.join(VOC_PATH, 'label_map.pbtxt')

CONFIG_PATH = os.path.join(PIPELINE_PATH, 'pipeline.config')

MODEL_PATH = os.path.join(PIPELINE_PATH, MODEL_DIR)
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')
EXPORT_PATH = os.path.join(MODEL_PATH, 'export')
PREDICTION_OUTPUT_PATH = os.path.join(MODEL_PATH, 'prediction')
