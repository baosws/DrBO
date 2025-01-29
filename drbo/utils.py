import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from castle.datasets import IIDSimulation, DAG
import numpy as np
import os

def sim_data(nodes, samples, edges_per_node, random_state=0):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=nodes * edges_per_node, weight_range=[.5, 2], seed=random_state)
    dataset = IIDSimulation(W=weighted_random_dag, n=samples, noise_scale=1, method='linear', sem_type='gauss')
    return dataset

def read_nonlinear_gp(name, id):
    data_path = f'data/{name}/data{id}.npy'
    dag_path = f'data/{name}/DAG{id}.npy'
    if not os.path.exists(data_path) or not os.path.exists(dag_path):
        raise RuntimeError('Data not found. Download the data at https://github.com/kurowasan/GraN-DAG/blob/master/data/data_p10_e40_n1000_GP.zip then, extract it to "data/"')
    data = np.load(data_path)
    dag = np.load(dag_path)
    return dict(X=data, GT=1 * (np.abs(dag) > 1e-3))

def read_sachs():
    data_path = 'data/sachs/observations.csv'
    dag_path = 'data/sachs/dag.csv'
    if not os.path.exists(data_path) or not os.path.exists(dag_path):
        raise RuntimeError(f'Sachs data not found. Requiring 2 files: "{data_path}" & "{dag_path}"')
    X = pd.read_csv(data_path).values
    GT = pd.read_csv(dag_path, index_col='source').values
    return dict(X=X, GT=GT)
    
def viz_history(df):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))
    sns.lineplot(df, x='step', y='BIC', ax=ax0)
    sns.lineplot(df, x='step', y='shd', ax=ax1)
    sns.lineplot(df, x='timestamp', y='shd', ax=ax2)
    fig.tight_layout()
    fig.show()