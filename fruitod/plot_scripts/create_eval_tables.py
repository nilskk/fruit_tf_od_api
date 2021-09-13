import glob
import os
from pathlib import Path
import pandas as pd


def generate_latex_files(save_dir, df):
    one_stage_precision = df[['Name',
                              'Precision/mAP',
                              'Precision/mAP@.50IOU',
                              'Precision/mAP@.75IOU',
                              'Precision/mAP (small)',
                              'Precision/mAP (medium)',
                              'Precision/mAP (large)']]
    one_stage_precision.rename({'Name': '\text{Model}',
                                'Precision/mAP': '$mAP$',
                                'Precision/mAP@.50IOU': '$AP_{50}$',
                                'Precision/mAP@.75IOU': '$AP_{75}$',
                                'Precision/mAP (small)': '$AP_{small}$',
                                'Precision/mAP (medium)': '$AP_{medium}$',
                                'Precision/mAP (large)': '$AP_{large}$'}, axis=1, inplace=True)
    with open(os.path.join(save_dir, 'precision.txt'), 'w') as f:
        f.write(one_stage_precision.to_latex(index=False, escape=False, column_format='S||S|S|S|S|S|S'))

    one_stage_precision_classes = df[['Name',
                                      'PerformanceByCategory/mAP/apfel',
                                      'PerformanceByCategory/mAP/kiwi',
                                      'PerformanceByCategory/mAP/Kohlrabi',
                                      'PerformanceByCategory/mAP/mango',
                                      'PerformanceByCategory/mAP/paprika_mix',
                                      'PerformanceByCategory/mAP/rote paprika',
                                      'PerformanceByCategory/mAP/tomate']]
    one_stage_precision_classes.rename({'Name': '\text{Model}',
                                        'PerformanceByCategory/mAP/apfel': '\text{Apfel}',
                                        'PerformanceByCategory/mAP/kiwi': '\text{Kiwi}',
                                        'PerformanceByCategory/mAP/Kohlrabi': '\text{Kohlrabi}',
                                        'PerformanceByCategory/mAP/mango': '\text{Mango}',
                                        'PerformanceByCategory/mAP/paprika_mix': '\shortstack{Paprika \\\ Mix}',
                                        'PerformanceByCategory/mAP/rote paprika': '\shortstack{Paprika \\\ Rot}',
                                        'PerformanceByCategory/mAP/tomate': '\text{Tomate}'}, axis=1, inplace=True)
    with open(os.path.join(save_dir, 'precision_classes.txt'), 'w') as f:
        f.write(one_stage_precision_classes.to_latex(index=False, escape=False, column_format='S||S|S|S|S|S|S|S'))

    one_stage_recall = df[['Name',
                           'Recall/AR@1',
                           'Recall/AR@10',
                           'Recall/AR@100',
                           'Recall/AR@100 (small)',
                           'Recall/AR@100 (medium)',
                           'Recall/AR@100 (large)']]
    one_stage_recall.rename({'Name': '\text{Model}',
                             'Recall/AR@1': '$AR_{1}$',
                             'Recall/AR@10': '$AR_{10}$',
                             'Recall/AR@100': '$AR_{100}$',
                             'Recall/AR@100 (small)': '$AR_{small}$',
                             'Recall/AR@100 (medium)': '$AR_{medium}$',
                             'Recall/AR@100 (large)': '$AR_{large}$'}, axis=1, inplace=True)
    with open(os.path.join(save_dir, 'recall.txt'), 'w') as f:
        f.write(one_stage_recall.to_latex(index=False, escape=False, column_format='S||S|S|S|S|S|S'))

    one_stage_recall_classes = df[['Name',
                                   'Recall/AR@100 ByCategory/apfel',
                                   'Recall/AR@100 ByCategory/kiwi',
                                   'Recall/AR@100 ByCategory/Kohlrabi',
                                   'Recall/AR@100 ByCategory/mango',
                                   'Recall/AR@100 ByCategory/paprika_mix',
                                   'Recall/AR@100 ByCategory/rote paprika',
                                   'Recall/AR@100 ByCategory/tomate']]
    one_stage_recall_classes.rename({'Name': '\text{Model}',
                                     'Recall/AR@100 ByCategory/apfel': '\text{Apfel}',
                                     'Recall/AR@100 ByCategory/kiwi': '\text{Kiwi}',
                                     'Recall/AR@100 ByCategory/Kohlrabi': '\text{Kohlrabi}',
                                     'Recall/AR@100 ByCategory/mango': '\text{Mango}',
                                     'Recall/AR@100 ByCategory/paprika_mix': '\shortstack{Paprika \\\ Mix}',
                                     'Recall/AR@100 ByCategory/rote paprika': '\shortstack{Paprika \\\ Rot}',
                                     'Recall/AR@100 ByCategory/tomate': '\text{Tomate}'}, axis=1, inplace=True)
    with open(os.path.join(save_dir, 'recall_classes.txt'), 'w') as f:
        f.write(one_stage_recall_classes.to_latex(index=False, escape=False, column_format='S||S|S|S|S|S|S|S'))

    one_stage_performance = df[['Name',
                                'Precision/mAP',
                                'Precision/mAP@.50IOU',
                                'Precision/mAP@.75IOU',
                                'Recall/AR@100',
                                'Parameter',
                                'flops',
                                'time_per_image']]
    one_stage_performance.rename({'Name': '\text{Model}',
                                  'Precision/mAP': '$mAP$',
                                  'Precision/mAP@.50IOU': '$AP_{50}$',
                                  'Precision/mAP@.75IOU': '$AP_{75}$',
                                  'Recall/AR@100': '$AR_{100}$',
                                  'Parameter': '\text{Parameter}',
                                  'flops': '\text{FLOPs}',
                                  'time_per_image': '\shortstack{CPU \\\ Inferenz}'}, axis=1, inplace=True)
    with open(os.path.join(save_dir, 'performance.txt'), 'w') as f:
        f.write(one_stage_performance.to_latex(index=False, escape=False, column_format='S||S|S|S|S|S|S|S'))


def generate_complete_df(parent_path):
    df_list = []
    for directory in os.listdir(parent_path):
        dir_paths = glob.glob(os.path.join(parent_path, directory, '*'))
        for dir_path in dir_paths:
            if '.idea' not in dir_path:
                name_params_path = os.path.join(dir_path, 'metrics.csv')
                metrics_path = os.path.join(dir_path, 'inference', 'metrics.csv')
                df1 = pd.read_csv(name_params_path)
                name_params_df = df1[['Name', 'Parameter']]
                metrics_df = pd.read_csv(metrics_path)

                metrics_df.reset_index(drop=True, inplace=True)
                metrics_df.drop(metrics_df.filter(regex="Unname"), axis=1, inplace=True)
                name_params_df.reset_index(drop=True, inplace=True)
                name_params_df.drop(name_params_df.filter(regex="Unname"), axis=1, inplace=True)

                whole_df = pd.concat([name_params_df, metrics_df], axis=1)
                df_list.append(whole_df)

    complete_df = pd.concat(df_list, axis=0)
    return complete_df


if __name__ == '__main__':
    one_stage_dir = Path(os.path.join(Path.home(), 'Desktop', 'one_stage'))
    two_stage_dir = Path(os.path.join(Path.home(), 'Desktop', 'two_stage'))
    # pd.set_option('display.max_columns', None)

    one_stage_df = generate_complete_df(one_stage_dir)
    two_stage_df = generate_complete_df(two_stage_dir)

    generate_latex_files(one_stage_dir, one_stage_df)
    generate_latex_files(two_stage_dir, two_stage_df)
