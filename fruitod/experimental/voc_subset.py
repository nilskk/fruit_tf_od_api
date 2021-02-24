import os
from pathlib import Path
from lxml import etree as ET
import shutil
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format

def make_label_map(classes, output_path):
    Path(output_path).mkdir(exist_ok=True, parents=True)

    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=1):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')

    with open(os.path.join(output_path, 'pascal_label_map.pbtxt'), 'w') as f:
        f.write(text)


def copy_annotations_and_images(classes, input_path, output_path):
    annotation_input_path = os.path.join(input_path, 'Annotations')
    print(annotation_input_path)
    annotation_output_path = os.path.join(output_path, 'Annotations')
    Path(annotation_output_path).mkdir(exist_ok=True, parents=True)

    image_input_path = os.path.join(input_path, 'JPEGImages')
    image_output_path = os.path.join(output_path, 'JPEGImages')
    Path(image_output_path).mkdir(exist_ok=True, parents=True)

    for xml in os.listdir(annotation_input_path):
        tree = ET.parse(os.path.join(annotation_input_path, xml))
        root = tree.getroot()

        if root.find('object') is not None:
            class_name = root.find('object').find('name').text
            if class_name in classes:
                shutil.copy2(os.path.join(annotation_input_path, xml), annotation_output_path)

                image_filename = root.find('filename').text
                shutil.copy2(os.path.join(image_input_path, image_filename), image_output_path)


def make_trainval_split(output_path):
    image_path = os.path.join(output_path, 'JPEGImages')
    trainval_output_path = os.path.join(output_path, 'ImageSets/Main')
    Path(trainval_output_path).mkdir(exist_ok=True, parents=True)

    val_list = []
    train_list = []
    for i, image in enumerate(os.listdir(image_path)):
        if i % 5 == 0:
            val_list.append(image + '\n')
        else:
            train_list.append(image + '\n')

    with open(os.path.join(trainval_output_path, 'vott_train.txt'), 'w') as f:
        f.writelines(train_list)

    with open(os.path.join(trainval_output_path, 'vott_val.txt'), 'w') as f:
        f.writelines(val_list)


if __name__ == '__main__':
    voc_input_path = Path('./data/voc_data')
    voc_output_path = Path('./data/voc_data_3classes')


    classes = ['Mango', 'Muskmelon', 'Tomato']

    make_label_map(classes, voc_output_path)
    copy_annotations_and_images(classes, voc_input_path, voc_output_path)
    make_trainval_split(voc_output_path)


