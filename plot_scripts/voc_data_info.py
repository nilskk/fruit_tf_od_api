from lxml import etree as ET
from pathlib import Path
import json
import os
import seaborn as sns
import pandas as pd

plot_path = Path("../../plots/kaggle")

classes_list = ['Apple1', 'Kiwi1', 'Mango', 'Muskmelon', 'Tomato']


def parse_voc(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    num_objects = len(root.findall('object'))

    count_list = []
    class_list = []
    for voc_class in classes_list:
        class_count = len(root.findall("./object[name=\'"+voc_class+"\']"))
        if class_count > 0:
            class_list.append(voc_class)
            count_list.append(class_count)

    return class_list, count_list, num_objects


def create_numobjects_graph(df_classes):
    plt = sns.barplot(data=df_classes, x=df_classes['classes'], y=df_classes['num_objects'])
    plt.set(xlabel='Classes', ylabel='Number of Objects per Image')
    plt.get_figure().savefig(os.path.join(plot_path, 'num_objects_per_class_barplot.png'))
    plt.get_figure().clf()

    plt = sns.stripplot(data=df_classes, x=df_classes['classes'], y=df_classes['num_objects'],
                        jitter=0.4, size=8, linewidth=1, alpha=0.5)
    plt.set(xlabel='Classes', ylabel='Number of Objects per Image', yticks=range(12))
    plt.get_figure().savefig(os.path.join(plot_path, 'num_objects_per_class_stripplot.png'))
    plt.get_figure().clf()


if __name__ == "__main__":
    plot_path.mkdir(parents=True, exist_ok=True)

    xml_folder = Path("../data/voc_data/Annotations")

    class_list_ges = []
    count_list_ges = []
    num_objects_list = []

    for file in os.listdir(xml_folder):
        xml_file = os.path.join(xml_folder, file)
        class_list, count_list, num_objects = parse_voc(xml_path=xml_file)
        class_list_ges.append(class_list)
        count_list_ges.append(count_list)
        num_objects_list.append(num_objects)

    class_count_list_ges = list(zip(class_list_ges, count_list_ges))

    print(class_list_ges)

    df_classes = pd.DataFrame(data=class_count_list_ges, columns=['classes', 'num_objects'])
    df_classes['classes'] = df_classes['classes'].str[0]
    df_classes['num_objects'] = df_classes['num_objects'].str[0]
    df_gesamt = pd.DataFrame(data=num_objects_list, columns=['num_objects_gesamt'])

    print(df_classes)

    create_numobjects_graph(df_classes=df_classes)
