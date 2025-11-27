import torch
import pandas as pd
import numpy as np
import time
import statistics
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------
# DEVICE SETUP
# --------------------------
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------
# LOAD DATA DIRECTLY (NO CSV NEEDED)
# --------------------------
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA', '?']
)

# --------------------------
# PREPROCESS DATA
# --------------------------
df = pd.concat([df, pd.get_dummies(df['job'], prefix="job", dtype=int)], axis=1)
df.drop('job', axis=1, inplace=True)

df = pd.concat([df, pd.get_dummies(df['area'], prefix="area", dtype=int)], axis=1)
df.drop('area', axis=1, inplace=True)

df['income'] = df['income'].fillna(df['income'].median())

# Standardization
for col in ['income', 'aspect', 'save_rate', 'age', 'subscriptions']:
    df[col] = zscore(df[col])

x_columns = df.columns.drop(['product', 'id'])
x = df[x_columns].values

dummies = pd.get_dummies(df['product'])
y = dummies.values

# Convert to tensors
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.LongTensor(np.argmax(y, axis=1)).to(device)

# --------------------------
# MODEL
# --------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout, neuronPct, neuronShrink):
        super().__init__()
        layers = []
        neuronCount = int(neuronPct * 5000)
        prev = input_dim
        layer = 0

        while neuronCount > 25 and layer < 10:
            layers.append(nn.Linear(prev, neuronCount))
            prev = neuronCount
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            neuronCount = int(neuronCount * neuronShrink)
            layer += 1

        layers.append(nn.Linear(prev, y.shape[1]))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --------------------------
# EVALUATION
# --------------------------
SPLITS = 2
EPOCHS = 500
PATIENCE = 10

def evaluate_network(learning_rate=1e-3, dropout=0.2, neuronPct=0.1, neuronShrink=0.25):
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1)
    mean_loss = []

    for train, test in boot.split(x, np.argmax(y, axis=1)):
        x_train = x_tensor[train]
        y_train = y_tensor[train]
        x_test = x_tensor[test]
        y_test = y_tensor[test]

        model = NeuralNetwork(x.shape[1], dropout, neuronPct, neuronShrink).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

        best_loss = 9999
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_test), y_test).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1

            if patience >= PATIENCE:
                break

        with torch.no_grad():
            pred = model(x_test).cpu().numpy()
            y_test_cpu = y_test.cpu().numpy()
            score = metrics.log_loss(y_test_cpu, pred)
            mean_loss.append(score)

    return -statistics.mean(mean_loss)

# --------------------------
# BAYESIAN OPTIMIZATION
# --------------------------
pbounds = {
    'dropout': (0.0, 0.499),
    'learning_rate': (0.0001, 0.1),
    'neuronPct': (0.01, 1),
    'neuronShrink': (0.01, 1)
}

optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    random_state=1,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=10)

print("\nBEST PARAMETERS FOUND:")
print(optimizer.max)
