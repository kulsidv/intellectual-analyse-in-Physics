import pandas as pd
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import combinations


class NeuroNet(torch.nn.Module):
    def __init__(self, input_nodes_n):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_nodes_n, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y, lr=0.01, epoches=100):
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(1)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        self.train()
        for epoch in range(epoches):
            optimizer.zero_grad()
            logits = self(X_tensor)
            loss_value = loss(logits, y_tensor)
            loss_value.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            if X_tensor.dim() == 1:
                X_tensor = X_tensor.unsqueeze(1)
            logits = self(X_tensor)
            return logits.numpy().flatten()


def vision(results):
    results = results.sort_values('Error', ascending=True)  # сортировка
    params = [', '.join(p) if isinstance(p, tuple) else str(p) for p in results['Params']]
    errors = results['Error']

    plt.figure(figsize=(10, 5))
    plt.barh(params, errors, edgecolor='black')  # горизонтальные бары
    plt.xlabel('MAE (ошибка)')
    plt.ylabel('Параметры')
    plt.title('Ошибка предсказания')
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("dataset.csv", sep=",")

    df = df[["Tg", "delta Cp(Tg)", "delta Hm", "delta Cp/delta Sm", "m_meas"]]

    normalizer = MaxAbsScaler()
    normalized_arr = normalizer.fit_transform(df)
    df_norm = pd.DataFrame(normalized_arr, columns=df.columns)

    results = []
    max_m_meas = df["m_meas"].max()
    for i in range(1, 5):
        for columns_set in combinations(
            ["Tg", "delta Cp(Tg)", "delta Hm", "delta Cp/delta Sm"], i
        ):
            X_train, X_test, y_train, y_test = train_test_split(
                df_norm[list(columns_set)],
                df_norm["m_meas"],
                test_size=0.2,
                random_state=42,
            )

            model = NeuroNet(i).fit(X_train, y_train, epoches=600)
            y_pred = model.predict(X_test)
            y_pred_orig = y_pred * max_m_meas
            y_test_orig = y_test * max_m_meas
            error = mean_absolute_error(y_test_orig, y_pred_orig)

            row = dict()
            row["Params"] = columns_set
            row["Error"] = error
            results.append(row)
    results = pd.DataFrame(results, columns=["Params", "Error"])

    print(results)

    vision(results)


if __name__ == "__main__":
    main()
