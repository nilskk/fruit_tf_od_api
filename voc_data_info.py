import xml.etree.ElementTree as ET
from pathlib import Path

def main():
    return

def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in list(root):
        print(child.get("key"))
        if child.get("name") == "object":
            print(child)

    return

if __name__=="__main__":
    voc_example = Path("./data/voc_data/Annotations/Apple%201.xml")

    parse_voc(voc_example)