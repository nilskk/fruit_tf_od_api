import shutil
import glob
import os
import urllib.parse
from pathlib import Path


if __name__ == '__main__':
    voc_path_string = '/data/voc_fruit_weights/'
    fruit_list = ['apfel', 'bunte_paprika', 'kiwi', 'kohlrabi', 'mango', 'paprika_rot', 'tomate']
    for fruit_name in fruit_list:
        input_path = '/data/classes/{}'.format(fruit_name)
        write_path = os.path.join(input_path, 'weights_correct_format')
        for file_path in glob.glob(os.path.join(input_path, 'weights/*')):
            if ' ' in file_path:
                file_name = Path(file_path).name
                file_name_percent_encoded = urllib.parse.quote(file_name)
                shutil.copy2(file_path, os.path.join(write_path, file_name_percent_encoded))
            else:
                shutil.copy2(file_path, write_path)