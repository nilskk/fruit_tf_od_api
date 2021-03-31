# fruit_tf_od_api

## Installation

1. Tensorflow Models Repository klonen
```bash
git clone https://github.com/nilskk/models.git
```

2. Object Detection Package installieren
```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

3. fruit_tf_od_api Repository klonen
```bash
git clone https://github.com/nilskk/fruit_tf_od_api.git
```

4. fruitod Package installieren
```bash
cd fruit_tf_od_api
python -m pip install --editable .
```


## Verwendung

1. Variablen in `fruitod/settings.py` anpassen
2. `fruitod/preprocess_data.py` ausführen
3. `fruitod/train_eval_export.py` ausführen
4. `fruitod/predict.py` ausführen

### settings.py Variablen

Variable        |   Bedeutung
---             |   ---
HOME            |   Pfad an dem alle Daten gespeichert werden sollen
VOC_PATH        |   Pfad zu dem VOC Ordner
TFRECORDS_PATH  |   Pfad an dem die generierten .tfrecord Dateien gespeichert werden sollen
MODEL_PATH      |   Pfad zum Ordner des aktuell verwendeten Models

### VOC Ordnerstruktur

```
<folder_name>
├── Annotations
│   ├── <image_1>.xml
│   ├── <image_2>.xml
│   ...
├── ImageSets
│   └── Main
│       ├── train.txt
│       └── val.txt
├── JPEGImages
│   ├── <image_1>.png
│   ├── <image_2>.png
│   ...
└── pascal_label_map.pbtxt
```

            


