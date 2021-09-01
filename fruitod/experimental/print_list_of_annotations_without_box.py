import glob

from lxml import etree as ET
from pathlib import Path
import json
import os
import numpy as np
import seaborn as sns
import pandas as pd
from lxml import etree

from object_detection.utils import dataset_util

if __name__ == '__main__':
    a_list = []
    for annotation_file in glob.glob('/data/classes/*/annotations/*.xml'):
        tree = etree.parse(annotation_file)
        count = tree.xpath('count(//object)')
        if count == 0:
            a_list.append(annotation_file)

    with open(os.path.join('/data/classes', 'annotations_without_box.txt'), 'w') as f:
        for element in a_list:
            f.write(str(element) + '\n')


