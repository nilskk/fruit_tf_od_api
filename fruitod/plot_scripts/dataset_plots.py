from lxml import etree as ET
from pathlib import Path
import json
import os
import seaborn as sns
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt

from object_detection.utils import dataset_util


if __name__ == '__main__':
    # data_dir_string = '/home/nilskk/rewe_project/data/voc_weight_test'
    data_dir_string = '/data/voc_fruit_weights'
    data_directory = Path(data_dir_string)
    input_directory = Path(os.path.join(data_dir_string, 'dataset_information'))
    output_directory = Path(os.path.join(data_dir_string, 'dataset_plots'))
    output_directory.mkdir(exist_ok=True, parents=True)

    file_dataframe = pd.read_pickle(os.path.join(input_directory, 'file_dataframe.pkl'))
    object_dataframe = pd.read_pickle(os.path.join(input_directory, 'object_dataframe.pkl'))

    sns.set_theme()

    # Anzahl Objekte pro Klasse Train/Test
    plt.figure()
    sns.catplot(data=file_dataframe, x='class', y='objects', hue='set', split=True,
                jitter=0.4, size=8, linewidth=1, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(os.path.join(output_directory, 'objects_per_class.png'), bbox_inches='tight')

    # Anzahl Bilder pro Klasse Train/Test
    plt.figure()
    ax = sns.countplot(data=file_dataframe, x='class', hue='set')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(os.path.join(output_directory, 'images_per_class.png'), bbox_inches='tight')

    # Anzahl Größe der Boxen
    plt.figure()
    sns.scatterplot(data=object_dataframe, x='width', y='height', hue='class', style='size', alpha=0.8)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(os.path.join(output_directory, 'box_size.png'), bbox_inches='tight')

    # Seitenverhältnisse der Boxen
    plt.figure()
    sns.catplot(data=object_dataframe, x='class', y='ratio', hue='set', split=True,
                jitter=0.4, size=8, linewidth=1, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(os.path.join(output_directory, 'box_ratio.png'), bbox_inches='tight')

    # Gewicht in Abhängigkeit von Anzahl der Objekte
    plt.figure()
    sns.lmplot(data=file_dataframe, x='weight', y='objects', hue='class')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(os.path.join(output_directory, 'weight_number_of_objects.png'), bbox_inches='tight')

