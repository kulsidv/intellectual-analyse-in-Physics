import torch
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import re



class NeuroNet(torch.nn.Module):
    def __init__(self, input_nodes, output_nodes, loss_func):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_nodes, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, output_nodes),
        )
        self.loss_func = loss_func

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y, lr=0.01, epoches=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X.copy())
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        losses = []
        self.train()
        for epoch in range(epoches):
            optimizer.zero_grad()
            logits = self(X_tensor)
            loss_value = self.loss_func(logits, y_tensor)
            loss_value.backward()
            optimizer.step()
            losses.append(loss_value.item())
        return self, losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self(X_tensor)
            return logits.numpy().ravel()


def count_upper(word: str):
    count = 0
    for letter in word:
        if letter.isupper():
            count += 1
    return count


def plot_loss(losses, save_path=None):
    """Отрисовка графика функции потерь"""
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2, color='steelblue')
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Динамика функции потерь', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)


def safe_filename(combo, ext=".png"):
    """Превращает кортеж признаков в безопасное имя файла"""
    # Склеиваем через "_", заменяем всё кроме букв, цифр и "_" на "_"
    name = "_".join(str(c) for c in combo)
    safe_name = re.sub(r'[^\w\-]', '_', name)
    # Убираем дублирующиеся подчёркивания
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    return safe_name.strip('_') + ext


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_excel("data_piezo.xlsx", sheet_name=0)

    unique_space_group = list(df["space group"].unique())
    unique_space_group_dict = dict(
        zip(unique_space_group, range(len(unique_space_group)))
    )
    df["space group class"] = df["space group"].map(unique_space_group_dict)

    df["materials count"] = df["formula"].str.count(r"[A-Z]").fillna(0)

    features = [
        "materials count",
        "space group class",
        "band gap (eV)",
        "density (g/cm^3)",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["piezoelectric modulus (C/m^2)"],
        test_size=0.2,
        random_state=42,
    )

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    for i in range(1, 5):
        for combo in combinations(features, i):
            list_combo = list(combo)
            X_train_arr = X_train[list_combo].to_numpy()
            X_test_arr = X_test[list_combo].to_numpy()
            model, losses = NeuroNet(i, 1, torch.nn.MSELoss()).fit(X_train_arr, y_train_arr)
            y_pred = model.predict(X_test_arr)
            mae = mean_absolute_error(y_test_arr, y_pred)
            print(f"MAE for combination {combo}: {mae}")
            plot_loss(losses, safe_filename(combo))

    '''
    выяснилось, что лучшая комбинация: 'space group class', 'band gap (eV)', 'density (g/cm^3)'
    '''



if __name__ == "__main__":
    main()
