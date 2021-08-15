import os
from pathlib import Path
import shutil
import glob
import json
import lxml.etree
import numpy as np
import urllib.parse


def _make_voc_directories(voc_base_path):
    Path(voc_base_path).mkdir(parents=True, exist_ok=True)
    trainval_path = os.path.join(voc_base_path, 'ImageSets/Main')
    Path(trainval_path).mkdir(parents=True, exist_ok=True)
    image_path = os.path.join(voc_base_path, 'JPEGImages')
    Path(image_path).mkdir(parents=True, exist_ok=True)
    annotation_path = os.path.join(voc_base_path, 'Annotations')
    Path(annotation_path).mkdir(parents=True, exist_ok=True)
    weight_path = os.path.join(voc_base_path, 'Weights')
    Path(weight_path).mkdir(parents=True, exist_ok=True)


def _create_train_val_lists(fruit_name):
    val_list =[]
    train_list = []
    for i, image_name in enumerate(glob.glob('/data/classes/{}/images/*'.format(fruit_name))):
        image_name_with_extension = Path(image_name).name
        image_name_without_extension = Path(image_name).stem
        json_path = Path(os.path.join('/data/classes/{}/weights_correct_format'.format(fruit_name), image_name_without_extension + '.json'))

        if not json_path.is_file():
            train_list.append(image_name_with_extension)
        else:
            with open(json_path) as f:
                json_dict = json.load(f)
            if 'weightInGrams' not in json_dict.keys():
                train_list.append(image_name_with_extension)
            elif i % 5 == 0:
                val_list.append(image_name_with_extension)
            else:
                train_list.append(image_name_with_extension)

    return train_list, val_list


def _copy_images_annotations_weights(fruit_name, voc_base_path):
    input_path = '/data/classes/{}'.format(fruit_name)
    for file_path in glob.glob(os.path.join(input_path, 'images/*')):
        shutil.copy2(file_path, os.path.join(voc_base_path, 'JPEGImages'))
    for file_path in glob.glob(os.path.join(input_path, 'annotations/*')):
        shutil.copy2(file_path, os.path.join(voc_base_path, 'Annotations'))
    for file_path in glob.glob(os.path.join(input_path, 'weights_correct_format/*')):
        shutil.copy2(file_path, os.path.join(voc_base_path, 'Weights'))


def _count_number_of_objects(fruit_name, xml_file_name):
    xml = lxml.etree.parse(
        os.path.join('/data/classes/{}/annotations'.format(fruit_name), xml_file_name))
    count = float(xml.xpath('count(//object)'))
    return count


def _save_weight_to_json(json_path, count, weight_per_object):
    weight = count * weight_per_object
    json_dict = {'weightInGrams': weight}
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)


def _create_missing_weights(fruit_name):
    weight_per_object_list = []
    # calculate mean weight per object
    weight_path = '/data/classes/{}/weights_correct_format'.format(fruit_name)
    for weight_file in glob.glob(os.path.join(weight_path, '*')):
        file_name_without_extension = Path(weight_file).stem
        with open(weight_file) as f:
            json_dict = json.load(f)
        if 'weightInGrams' in json_dict.keys():
            weight = float(json_dict['weightInGrams'])

            count = _count_number_of_objects(fruit_name, file_name_without_extension + '.xml')

            if count > 0:
                weight_per_object_list.append(float(weight/count))

    weight_per_object = np.mean(np.asarray(weight_per_object_list))

    for image_name in glob.glob('/data/classes/{}/images/*'.format(fruit_name)):
        image_name_without_extension = Path(image_name).stem
        json_path = Path(
            os.path.join('/data/classes/{}/weights_correct_format'.format(fruit_name), image_name_without_extension + '.json'))

        count = _count_number_of_objects(fruit_name, image_name_without_extension + '.xml')

        if not json_path.is_file():
            _save_weight_to_json(json_path, count, weight_per_object)
        else:
            with open(json_path) as f:
                json_dict = json.load(f)
            if 'weightInGrams' not in json_dict.keys():
                _save_weight_to_json(json_path, count, weight_per_object)


def _write_lists(list, txt_path):
    with open(txt_path, 'w') as f:
        f.writelines('\n'.join(list))


if __name__ == '__main__':
    voc_path_string = '/data/voc_fruit_weights/'
    _make_voc_directories(voc_path_string)

    fruit_list = ['apfel', 'bunte_paprika', 'kiwi', 'kohlrabi','mango', 'paprika_rot', 'tomate']
    train_list_gesamt = []
    val_list_gesamt = []
    for fruit_name in fruit_list:
        train_list, val_list = _create_train_val_lists(fruit_name)
        for name in train_list:
            train_list_gesamt.append(name)
        for name in val_list:
            val_list_gesamt.append(name)

        _write_lists(train_list_gesamt, '/data/voc_fruit_weights/ImageSets/Main/train.txt')
        _write_lists(val_list_gesamt, '/data/voc_fruit_weights/ImageSets/Main/val.txt')

        _copy_images_annotations_weights(fruit_name, voc_path_string)

        _create_missing_weights(fruit_name)

