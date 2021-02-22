import pandas as pd
import os


def create_dataframe(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)
  path = os.path.join(dir, 'metrics.csv')
  df = pd.DataFrame(columns=['Name', 'Optimizer', 'Batch Size', 'Learning Rate', 'Parameter', 'Flops', 'Inference Speed',
                   'DetectionBoxes_Precision/mAP', 'Loss/total_loss'])
  df.to_csv(path, index=False)

def write_coco_metrics(dir, metrics):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  for k in metrics:
    df = df.drop(columns=k, axis=1, errors='ignore')
    df.insert(loc=0, column=k, value=[metrics[k]])
  df.to_csv(path, index=False)

def write_metric(dir, metric_name, metric_value):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns=metric_name, axis=1, errors='ignore')
  df.insert(loc=0, column=metric_name, value=[metric_value])
  df.to_csv(path, index=False)
