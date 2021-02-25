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