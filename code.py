
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os


window_size = 31
prediction_horizon = 1
hidden_size = 33

print("Window Size:", window_size)
print("Horizon:", prediction_horizon)
print("Hidden Size:", hidden_size)


def create_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])
    return np.array(X), np.array(y)


class MLP(nn.Module):
    def __init__(self, input_size, horizon):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.hidden_size = hidden_size

        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_hat = nn.Linear(input_size + hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size)

        for t in range(x.size(1)):
            combined = torch.cat((x[:, t], h), dim=1)

            z = torch.sigmoid(self.z(combined))
            r = torch.sigmoid(self.r(combined))

            combined_r = torch.cat((x[:, t], r * h), dim=1)
            h_tilde = torch.tanh(self.h_hat(combined_r))

            h = (1 - z) * h + z * h_tilde

        return self.fc(h)

def train(model, X, y, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return losses, model

def run_pipeline(data, dataset_name):
    print(f"\n===== Running on {dataset_name} =====")

    data = (data - data.mean()) / data.std()

    X, y = create_windows(data, window_size, prediction_horizon)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    gru = CustomGRU(1, hidden_size, prediction_horizon)

    losses, gru = train(gru, X_train, y_train)

    # Plot Loss
    plt.figure()
    plt.plot(losses)
    plt.title(f"{dataset_name} - Training Loss")
    plt.savefig(f"/kaggle/working/{dataset_name}_loss.png")
    plt.show()

    # Prediction
    pred = gru(X_test).detach().numpy()
    actual = y_test.numpy()

    plt.figure()
    plt.plot(actual[:100, 0], label="Actual")
    plt.plot(pred[:100, 0], label="Predicted")
    plt.legend()
    plt.title(f"{dataset_name} - Prediction")
    plt.savefig(f"/kaggle/working/{dataset_name}_prediction.png")
    plt.show()

    # Metrics
    mse = ((pred - actual)**2).mean()
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mse)

    print(f"{dataset_name} Results:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)


file_path = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv') or filename.endswith('.txt'):
            file_path = os.path.join(dirname, filename)

print("Using file:", file_path)

if file_path.endswith('.txt'):
    df = pd.read_csv(file_path, sep=';', low_memory=False)
else:
    df = pd.read_csv(file_path)

data1 = df.iloc[:, 1].dropna().values


try:
    df2 = pd.read_csv("/kaggle/input/stock-data/stock.csv")
    data2 = df2['Close'].dropna().values
except:
    print("Stock dataset not found — using synthetic data instead")
    data2 = np.sin(np.linspace(0, 100, 500))


run_pipeline(data1, "Electricity")
run_pipeline(data2, "Stock")
