from lxml import etree as ET
from pathlib import Path
import json
import os
import seaborn as sns
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt

from object_detection.utils import dataset_util


def _read_weight_complete(weight_file):
    with open(weight_file) as f:
        json_dict = json.load(f)
    weight_complete = json_dict['weightInGrams']
    return weight_complete


def _get_annotation_information(annotation_file, weight_complete):
    tree = etree.parse(annotation_file)
    number_of_objects = int(tree.xpath('count(//object)'))
    if number_of_objects > 0:
        class_label = tree.find('object').find('name').text
    else:
        class_label = 'unspecified'

    single_file_dict = {'objects': number_of_objects,
                        'class': class_label}

    object_list = []
    for object in tree.findall('object'):
        bndbox_class = object.find('name').text
        xmin = float(object.find('bndbox').find('xmin').text)
        xmax = float(object.find('bndbox').find('xmax').text)
        ymin = float(object.find('bndbox').find('ymin').text)
        ymax = float(object.find('bndbox').find('ymax').text)
        bndbox_width = xmax - xmin
        bndbox_height = ymax - ymin
        bndbox_area = bndbox_height * bndbox_width
        bndbox_ratio = bndbox_width / bndbox_height
        bndbox_size = ''
        if bndbox_area < 32**2:
            bndbox_size = 'small'
        elif bndbox_area < 96**2:
            bndbox_size = 'medium'
        else:
            bndbox_size = 'large'

        weight_per_object = float(weight_complete / number_of_objects)

        object_dict = {'class': bndbox_class,
                       'width': bndbox_width,
                       'height': bndbox_height,
                       'ratio': bndbox_ratio,
                       'area': bndbox_area,
                       'size': bndbox_size,
                       'weight': weight_per_object}
        object_list.append(object_dict)

    return single_file_dict, object_list


if __name__ == '__main__':
    # data_dir_string = '/home/nilskk/rewe_project/data/voc_weight_test'
    data_dir_string = '/data/voc_fruit_weights'
    data_directory = Path(data_dir_string)
    output_directory = Path(os.path.join(data_dir_string, 'dataset_information'))
    output_directory.mkdir(exist_ok=True, parents=True)

    train_file = os.path.join(data_directory, 'ImageSets', 'train.txt')
    train_list = dataset_util.read_examples_list(train_file)
    test_file = os.path.join(data_directory, 'ImageSets', 'test.txt')
    test_list = dataset_util.read_examples_list(test_file)

    # read information per file and per object from annotation and weight file and save to dict
    file_data_list = []
    object_data_list = []
    for dataset, set_name in zip([train_list, test_list], ['train', 'test']):
        for example in dataset:
            example_without_extension = Path(example).stem

            weight_file = os.path.join(data_directory, 'Weights', example_without_extension + '.json')
            weight_complete = _read_weight_complete(weight_file)

            annotation_file = os.path.join(data_directory, 'Annotations', example_without_extension + '.xml')
            single_file_dict, object_list = _get_annotation_information(annotation_file, weight_complete)

            file_dict = {'filename': example_without_extension,
                         'objects': single_file_dict['objects'],
                         'class': single_file_dict['class'],
                         'weight': weight_complete,
                         'set': set_name}
            file_data_list.append(file_dict)

            for object_dict in object_list:
                object_dict['set'] = set_name
                object_dict['filename'] = example_without_extension
            object_data_list = [*object_data_list, *object_list]

    file_dataframe = pd.DataFrame(file_data_list)
    file_dataframe.to_pickle(os.path.join(output_directory, 'file_dataframe.pkl'))
    object_dataframe = pd.DataFrame(object_data_list)
    object_dataframe.to_pickle(os.path.join(output_directory, 'object_dataframe.pkl'))
    print(file_dataframe)
    print(object_dataframe)
