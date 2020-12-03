import pandas as pd
import os


def create_dataframe(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)
  path = os.path.join(dir, 'metrics.csv')
  df = pd.DataFrame(list())
  df.to_csv(path, index=False)

def write_name(dir, name):
  if not os.path.exists(dir):
    os.mkdir(dir)
  path = os.path.join(dir, 'metrics.csv')
  df = pd.DataFrame([])
  df = df.drop(columns='Name', axis=1, errors='ignore')
  df.insert(loc=0, column='Name', value=[name])
  df.to_csv(path, index=False)

def write_lr(dir, lr):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Learning Rate', axis=1, errors='ignore')
  df.insert(loc=0, column='Learning Rate', value=[lr])
  df.to_csv(path, index=False)

def write_bs(dir, bs):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Batch Size', axis=1, errors='ignore')
  df.insert(loc=0, column='Batch Size', value=[bs])
  df.to_csv(path, index=False)

def write_params(dir, params):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Parameter', axis=1, errors='ignore')
  df.insert(loc=0, column='Parameter', value=[params])
  df.to_csv(path, index=False)

def write_flops(dir, flops):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Flops', axis=1, errors='ignore')
  df.insert(loc=0, column='Flops', value=[flops])
  df.to_csv(path, index=False)

def write_speed(dir, speed):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Inference Speed', axis=1, errors='ignore')
  df.insert(loc=0, column='Inference Speed', value=[speed])
  df.to_csv(path, index=False)

def write_metrics(dir, metrics):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  for k in metrics:
    df = df.drop(columns=k, axis=1, errors='ignore')
    df.insert(loc=0, column=k, value=[metrics[k]])
  df.to_csv(path, index=False)

def write_optimizer(dir, optimizer_name):
  path = os.path.join(dir, 'metrics.csv')
  df = pd.read_csv(path)
  df = df.drop(columns='Optimizer', axis=1, errors='ignore')
  df.insert(loc=0, column='Optimizer', value=[optimizer_name])
  df.to_csv(path, index=False)
