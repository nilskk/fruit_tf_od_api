import os


TFRECORDS_PATH = './data/tfrecords'
TRAIN_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'vott_train.tfrecord')
VAL_TFRECORD_PATH = os.path.join(TFRECORDS_PATH, 'vott_val.tfrecord')

VOC_PATH = './data/voc_data'
LABELMAP_PATH = os.path.join(VOC_PATH, 'pascal_label_map.pbtxt')
VOC_SET_NAME = 'vott'

MODEL_PATH = './models/test/efficientdet_d0'
CONFIG_PATH = os.path.join(MODEL_PATH, 'pipeline.config')

MODEL_NAME = os.path.basename(MODEL_PATH)
OPTIMIZER_NAME = 'adam'
NUM_CLASSES = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 10
FIRST_DECAY_EPOCHS = 10

SAVE_PATH = os.path.join(MODEL_PATH,
                         'lr={}_bs={}_classes={}_{}'.format(LEARNING_RATE,
                                                            BATCH_SIZE,
                                                            NUM_CLASSES,
                                                            OPTIMIZER_NAME))
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints')
EXPORT_PATH = os.path.join(SAVE_PATH, 'export')

PREDICTION_OUTPUT_PATH = os.path.join(SAVE_PATH, 'prediction')
SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.95
VISUALIZE = False
