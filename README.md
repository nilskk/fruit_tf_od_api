# fruit_tf_od_api

## Installation with Docker

1. Download Dockerfile from `dockerfiles` directory
2. Build Docker Image with the following command
```
docker run -it --gpus all -v <data_path>:/data -v <code_path>:/code fruit_tf_od_api:1.0
```
- `<data_path>` should be a path to a directory on your machine, where you want to save the VOC Dataset directory and model directories
- `<code_path>` is the path, where the Dockerfile will save the downloaded Github Repositories

---

## Prerequisites
The following prerequisites must be fulfilled for the scripts to work:

### VOC Dataset Structure
The dataset must be in this format, based on VOC to be read correct from the code.

```
<VOC_DIR>
 ├── Annotations
 │   ├── <image_1>.xml
 │   ├── <image_2>.xml
 │   └── ...
 ├── ImageSets
 │   ├── train.txt
 │   └── test.txt
 ├── JPEGImages
 │   ├── <image_1>.png
 │   ├── <image_2>.png
 │   └── ...
 ├── Weights (optional)
 │   ├── <image_1>.json
 │   ├── <image_2>.json
 │   └── ...
 └── label_map.pbtxt
```

### Model Structure
Copy one of the `pipeline.config` files from `model_configs` directory to a subfolder in your `data` directory, so that it looks like this:

```
<PIPELINE_DIR>
 └── pipeline.config
```

---

## Tutorial

1. Change variables in `fruitod/settings_gpu_0.py` or `fruitod/settings_gpu_1.py` depending on the graphics card to use
<br/><br/>
2. Create `train_gpu_{0,1}.tfrecord`, `test_gpu_{0,1}.tfrecord` and `scaler_minmax.pkl` in the dataset directory with
```bash
python fruitod/create_tfrecord_from_voc.py --gpu {0,1}
```
Now the VOC dataset directory should look like this:
```
<VOC_DIR>
 ├── Annotations
 │   ├── <image_1>.xml
 │   ├── <image_2>.xml
 │   └── ...
 ├── ImageSets
 │   ├── train.txt
 │   └── test.txt
 ├── JPEGImages
 │   ├── <image_1>.png
 │   ├── <image_2>.png
 │   └── ...
 ├── Weights (optional)
 │   ├── <image_1>.json
 │   ├── <image_2>.json
 │   └── ...
 ├── label_map.pbtxt
 ├── minmax_scaler.pkl
 ├── train_gpu_{0,1}.tfrecord
 └── test_gpu_{0,1}.tfrecord
```

<br/><br/>
3. Train, Evaluate and Export Model with 
```bash
python fruitod/train_eval_export.py --gpu {0,1}
```

Now the directory with the `pipeline.config` should look like this:
```
<PIPELINE_DIR>
 ├── <MODEL_DIR>
 │   ├── checkpoints
 │   └── export
 └── pipeline.config
```
<br/><br/>
3. Collect Metrics of the exported model on the test set with 
```
python fruitod/calculate_metrics.py
  --labelmap_file: Name of labelmap File
    (default: 'label_map.pbtxt')
  --model_path: Path to model (has checkpoints directory)
  --scaler_file: Filename of scaler pickle
    (default: 'minmax_scaler.pkl')
  --set_file: Name of file with images list to evaluate
    (default: 'test.txt')
  --[no]side_input: wether to include weights as side input into model
    (default: 'false')
  --voc_path: Path to voc dataset folder
```
Now the directory with the `pipeline.config` should look like this:
```
<PIPELINE_DIR>
 ├── <MODEL_DIR>
 │   ├── checkpoints
 │   ├── export
 │   └── metrics
 └── pipeline.config
```
<br/><br/>
4. Predict new images from a directory with
```
python fruitod/predict_images.py:
  --images_path: Path to folder with images for prediction
  --iou_threshold: IoU threshold for extra class agnostic NMS
    (default: '0.95')
  --labelmap_file: Name of labelmap File
    (default: 'label_map.pbtxt')
  --model_path: Path to model (has checkpoints directory)
  --scaler_file: Filename of scaler pickle
    (default: 'minmax_scaler.pkl')
  --score_threshold: score threshold for extra class agnostic NMS and printing
    (default: '0.5')
  --[no]side_input: wether to include weights as side input into model
    (default: 'false')
  --voc_path: Path to voc dataset folder with labelmap and tfrecord and optional weights and scaler file
  --weight_path: Path to directory with weight json files
```

The predictions are saved in a folder `prediction` in the images directory, including images with boxes and new VOC Annotations based on the predictions. 

---
## Custom Tensorflow Object Detection API

Link: https://github.com/nilskk/models/tree/master/research/object_detection




            


