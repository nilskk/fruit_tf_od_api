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
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=40, ha="right")
    # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Objekte pro Bild')
    plt.savefig(os.path.join(output_directory, 'objects_per_class.png'), bbox_inches='tight')

    # Anzahl Bilder pro Klasse Train/Test
    plt.figure(figsize=(8,8))
    ax = sns.countplot(data=file_dataframe, x='class', hue='set')
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=40, ha="right")
    # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Bilder')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.savefig(os.path.join(output_directory, 'images_per_class.png'), bbox_inches='tight')

    # Anzahl Größe der Boxen
    plt.figure(figsize=(8,8))
    markers = {"medium": "s", "large": "X"}
    ax = sns.relplot(data=object_dataframe, x='width', y='height', col='class', col_wrap=3, hue='size')
    # sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabels('Boxweite in Pixel', fontsize=15)  # not set_label
    ax.set_ylabels('Boxhöhe in Pixel', fontsize=15)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # plt.xlabel('Boxweite in Pixel')
    # plt.ylabel('Boxhöhe in Pixel')
    plt.savefig(os.path.join(output_directory, 'box_size.png'), bbox_inches='tight')

    # Fläche der Boxen
    plt.figure(figsize=(9,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='area')
    # sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.axhline(96**2, ls='--', c='blue')
    ax.axhline(32**2, ls='--', c='orange')
    plt.xticks(rotation=40, ha="right")
    # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Fläche in Pixelanzahl')
    plt.savefig(os.path.join(output_directory, 'box_area.png'), bbox_inches='tight')

    # Seitenverhältnisse der Boxen
    plt.figure(figsize=(9,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='ratio')
    ax.set_yscale('log')
    # sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    locs = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    ax.yaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=40, ha="right")
    # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Klasse')
    plt.ylabel('Seitenverhältnis Weite / Höhe')
    plt.savefig(os.path.join(output_directory, 'box_ratio.png'), bbox_inches='tight')

    # Gewicht in Abhängigkeit von Anzahl der Objekte
    plt.figure(figsize=(8,8))
    g = sns.catplot(data=file_dataframe, y='weight', x='objects', hue='objects', col='class', col_wrap=4, jitter=0.4, alpha=0.8)
    for ax in g.axes.flat:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    g.set_ylabels('Gewicht in Gramm', fontsize=15)  # not set_label
    g.set_xlabels('Anzahl der Objekte', fontsize=15)
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
    plt.figure(figsize=(16,8))
    ax = sns.boxenplot(data=object_dataframe, x='class', y='weight', showfliers=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('Klasse')
    plt.ylabel('Gesamtgewicht / Anzahl der Objekte')
    plt.xticks(rotation=40, ha="right")

    plt.savefig(os.path.join(output_directory, 'weight_pro_object_wo_outlier.png'), bbox_inches='tight')

