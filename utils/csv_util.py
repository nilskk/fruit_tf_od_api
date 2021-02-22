import pandas as pd
import os
from pathlib import Path


def write_metrics(dir, metrics):
    path = Path(os.path.join(dir, 'metrics.csv'))
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=metrics.keys())
    for k in metrics:
        df = df.drop(columns=k, axis=1, errors='ignore')
        df.insert(loc=0, column=k, value=[metrics[k]])
    df.to_csv(path, index=False)


def write_metric(dir, metric_name, metric_value):
    path = Path(os.path.join(dir, 'metrics.csv'))
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=metric_name)
    df = df.drop(columns=metric_name, axis=1, errors='ignore')
    df.insert(loc=0, column=metric_name, value=[metric_value])
    df.to_csv(path, index=False)
