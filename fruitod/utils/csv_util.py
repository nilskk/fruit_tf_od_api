import pandas as pd
import os
from pathlib import Path


def write_metrics(dir, metrics):
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = Path(os.path.join(dir, 'metrics.csv'))
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=metrics.keys())
    for k in metrics:
        df = df.drop(columns=k, axis=1, errors='ignore')
        df.insert(loc=0, column=k, value=[metrics[k]])
    df.to_csv(path, index=False)

def write_loss_to_csv(dir, loss_dict):
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = Path(os.path.join(dir, 'training.csv'))
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=loss_dict.keys())
    loss_df = pd.DataFrame(data=[loss_dict], columns=loss_dict.keys())
    df_write = pd.concat([df, loss_df])
    df_write.to_csv(path, index=False)

def write_eval_to_csv(dir, eval_dict):
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = Path(os.path.join(dir, 'evaluation.csv'))
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=eval_dict.keys())
    eval_df = pd.DataFrame(data=[eval_dict], columns=eval_dict.keys())
    df_write = pd.concat([df, eval_df])
    df_write.to_csv(path, index=False)