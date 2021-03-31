from fruitod.core.predict_image import predict
from fruitod.settings import *

if __name__ == '__main__':
    #inference(export_path=EXPORT_PATH,
    #          tfrecord_path=VAL_TFRECORD_PATH)

    #flops(export_path=EXPORT_PATH)

    #collect_csv(models_path=os.path.split(MODEL_PATH)[0])

    predict(export_path=EXPORT_PATH,
            output_path=PREDICTION_OUTPUT_PATH,
            labelmap_path=LABELMAP_PATH,
            tfrecord_path=VAL_TFRECORD_PATH,
            score_threshold=SCORE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            visualize=VISUALIZE)
