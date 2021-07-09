import shutil
import glob
import os
import urllib.parse


if __name__ == '__main__':
    voc_path_string = '/data/voc_fruit_weights/'
    fruit_list = ['apfel', 'bunte_paprika', 'kiwi', 'kohlrabi', 'mango', 'paprika_rot', 'tomate']
    for fruit_name in fruit_list:
        input_path = '/data/classes/{}'.format(fruit_name)
        for file_path in glob.glob(os.path.join(input_path, 'weights/*')):
            if ' ' in file_path:
                file_path = urllib.parse.quote(file_path)
            shutil.copy2(file_path, os.path.join(input_path, 'weights_correct_format'))