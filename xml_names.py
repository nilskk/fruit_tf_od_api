import xml.etree.ElementTree as ET
import os
from pathlib import Path

if __name__ == '__main__':
    xml_path = Path("/data/voc_data_penny/Annotations")
    for xml in os.listdir(xml_path):
        tree = ET.parse(os.path.join(xml_path, xml))
        root = tree.getroot()

        for object in root.findall('object'):
            print(object.find('name').text)


