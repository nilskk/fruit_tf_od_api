import glob

from lxml import etree as ET
from pathlib import Path
import json
import os
import numpy as np
import seaborn as sns
import pandas as pd
from lxml import etree

if __name__ == '__main__':

    missing_weight_files_list = []
    missing_entries_in_files_list = []

    fruit_list = ['apfel', 'bunte_paprika', 'kiwi', 'kohlrabi', 'mango', 'paprika_rot', 'tomate']
    for fruit_name in fruit_list:
        input_path = '/data/classes/{}'.format(fruit_name)

        for i, image_name in enumerate(glob.glob('/data/classes/{}/images/*'.format(fruit_name))):
            image_name_with_extension = Path(image_name).name
            image_name_without_extension = Path(image_name).stem
            json_path = Path(os.path.join('/data/classes/{}/weights'.format(fruit_name), image_name_without_extension + '.json'))

            if not json_path.is_file():
                missing_weight_files_list.append(image_name)
            else:
                with open(json_path) as f:
                    json_dict = json.load(f)
                if 'weightInGrams' not in json_dict.keys():
                    missing_entries_in_files_list.append(json_path)

    with open(os.path.join('/data/classes', 'images_with_missing_weight_files.txt'), 'w') as f:
        for element in missing_weight_files_list:
            f.write(str(element) + '\n')

    with open(os.path.join('/data/classes', 'missing_entries_in_weight_files.txt'), 'w') as f:
        for element in missing_entries_in_files_list:
            f.write(str(element) + '\n')
