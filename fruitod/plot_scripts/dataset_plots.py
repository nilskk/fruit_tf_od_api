from lxml import etree as ET
from pathlib import Path
import json
import os
import numpy as np
import seaborn as sns
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from object_detection.utils import dataset_util


if __name__ == '__main__':
    data_dir_string = '/home/nilskk/rewe_project/data/penny'
    # data_dir_string = '/data/voc_fruit_weights'
    data_directory = Path(data_dir_string)
    input_directory = Path(os.path.join(data_dir_string, 'dataset_information'))
    output_directory = Path(os.path.join(data_dir_string, 'dataset_plots'))
    output_directory.mkdir(exist_ok=True, parents=True)

    file_dataframe = pd.read_pickle(os.path.join(input_directory, 'file_dataframe.pkl'))
    object_dataframe = pd.read_pickle(os.path.join(input_directory, 'object_dataframe.pkl'))

    sns.set_theme()

    # Anzahl Objekte pro Klasse Train/Test
    plt.figure(figsize=(8,8))
    ax = sns.boxenplot(data=file_dataframe, x='class', y='objects', hue='set')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=40, ha="right")
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Objekte pro Bild')
    plt.savefig(os.path.join(output_directory, 'objects_per_class.png'), bbox_inches='tight')

    # Anzahl Bilder pro Klasse Train/Test
    plt.figure(figsize=(8,8))
    ax = sns.countplot(data=file_dataframe, x='class', hue='set')
    plt.xticks(rotation=40, ha="right")
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Bilder')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.savefig(os.path.join(output_directory, 'images_per_class.png'), bbox_inches='tight')

    # Anzahl Größe der Boxen
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=object_dataframe, x='width', y='height', hue='class', style='size', alpha=0.8)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Boxweite in Pixel')
    plt.ylabel('Boxhöhe in Pixel')
    plt.savefig(os.path.join(output_directory, 'box_size.png'), bbox_inches='tight')

    # Seitenverhältnisse der Boxen
    plt.figure(figsize=(8,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='ratio', hue='set')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=40, ha="right")
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Seitenverhältnis Weite / Höhe')
    plt.savefig(os.path.join(output_directory, 'box_ratio.png'), bbox_inches='tight')

    # Gewicht in Abhängigkeit von Anzahl der Objekte
    plt.figure(figsize=(8,8))
    sns.stripplot(data=file_dataframe, y='weight', x='objects', hue='class', jitter=0.4, alpha=0.8)
    plt.ylabel('Gewicht in Gramm')
    plt.xlabel('Anzahl der Objekte')
    plt.savefig(os.path.join(output_directory, 'weight_number_of_objects.png'), bbox_inches='tight')

    # Gewicht pro Objekt
    plt.figure(figsize=(8,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='weight')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('Klasse')
    plt.ylabel('Gesamtgewicht / Anzahl der Objekte')
    plt.xticks(rotation=40, ha="right")

    plt.savefig(os.path.join(output_directory, 'weight_pro_object.png'), bbox_inches='tight')

    # Gewicht pro Objekt
    plt.figure(figsize=(8,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='weight', showfliers=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('Klasse')
    plt.ylabel('Gesamtgewicht / Anzahl der Objekte')
    plt.xticks(rotation=40, ha="right")

    plt.savefig(os.path.join(output_directory, 'weight_pro_object_wo_outlier.png'), bbox_inches='tight')

