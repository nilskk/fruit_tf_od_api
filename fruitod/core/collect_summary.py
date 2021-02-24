import pandas as pd
import glob
from absl import flags, app
import os


flags.DEFINE_string('models_path', None, 'Path to models directory')

FLAGS = flags.FLAGS


def collect_csv(models_path):
    cols_to_order = ['Name', 'Optimizer', 'Batch Size', 'Learning Rate', 'Parameter', 'Flops', 'Inference Speed',
                     'DetectionBoxes_Precision/mAP', 'Loss/total_loss']

    files = glob.glob(models_path + '/**/metrics.csv', recursive=True)
    df_list = []
    for file in files:
        tmp_df = pd.read_csv(file)
        new_columns = cols_to_order + (tmp_df.columns.drop(cols_to_order).tolist())
        tmp_df = tmp_df[new_columns]
        df_list.append(tmp_df)

    summary_df = pd.concat(df_list, axis=0)
    summary_path = os.path.join(models_path, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)


def main(argv):
    flags.mark_flag_as_required('models_path')

    collect_csv(models_path=FLAGS.models_path)


if __name__ == '__main__':
    app.run(main)


