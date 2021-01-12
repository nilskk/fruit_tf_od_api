import xml.etree.ElementTree as ET
from pathlib import Path
import json
import os
import seaborn as sns

plot_path = Path("./plots/kaggle")

classes_list = ['Apple1', 'Kiwi1', 'Mango', 'Muskmelon', 'Tomato']

def get_classnumber_from_xml(root, voc_class):
    return len(root.findall("./object[name=\'"+voc_class+"\']"))

def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    num_objects = len(root.findall('object'))
    class_objects_dict = {"num_objects": num_objects}
    for voc_class in classes_list:
        class_count = get_classnumber_from_xml(root=root, voc_class=voc_class)
        class_objects_dict[voc_class] = class_count

    return class_objects_dict

def calculate_imagenum_per_numobjects(dict_list):
    imagenum_per_numobjects = {}
    for dict in dict_list:
        if dict['num_objects'] in imagenum_per_numobjects:
            imagenum_per_numobjects[dict['num_objects']] += 1
        else:
            imagenum_per_numobjects[dict['num_objects']] = 1

    return imagenum_per_numobjects

def create_numobjects_graph(dict):
    fig1 = plt.figure()
    fig1.tight_layout()
    plt.bar(*zip(*sorted(dict.items())))
    plt.savefig(os.path.join(plot_path, 'num_objects_gesamt.png'))
    


if __name__=="__main__":
    plot_path.mkdir(parents=True, exist_ok=True)

    xml_folder = Path("./data/voc_data/Annotations")

    dict_list = []
    for file in os.listdir(xml_folder):
        xml_file = os.path.join(xml_folder, file)
        class_dict = parse_voc(xml_path=xml_file)
        dict_list.append(class_dict)

    imagenum_per_numobjects = calculate_imagenum_per_numobjects(dict_list=dict_list)
    print(imagenum_per_numobjects)

    create_numobjects_graph(imagenum_per_numobjects)
