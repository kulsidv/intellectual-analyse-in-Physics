import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        self.train()
        for epoch in range(epoches):
            optimizer.zero_grad()
            logits = self(X_tensor)
            loss_value = self.loss_func(logits, y_tensor)
            loss_value.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self(X_tensor)
            return logits.numpy()


def read(mode, count=59):
    dataframes = []
    for i in range(1, count + 1):
        rows = []
        title = f"{mode+'s'}/Dist_{mode}_{i}.txt"
        with open(title, "r", encoding="UTF-8") as file:
            lines = file.readlines()
            for line in lines[17:]:
                parts = line.split()
                row = {
                    "Size": float(parts[0].replace(",", ".")),
                    "Amount": int(parts[1]),
                }
                rows.append(row)
            dataframes.append(pd.DataFrame(rows))
    return dataframes


def prepare(dfs, materials):
    dfs_len = min(len(df) for df in dfs)  # безопасная обрезка
    
    data = []
    for i in range(59):
        # начинаем с целевых переменных
        row = [
            materials.iloc[i]["Модуль Юнга"],
            materials.iloc[i]["Предел прочности"]
        ]
        # добавляем Amount строго по позиции j (нули уже в файлах, если нужно)
        for j in range(dfs_len):
            row.append(dfs[i].iloc[j]["Amount"])
        data.append(row)
        
    # колонки: первые две фиксированные, остальные берём из первого файла
    feature_names = [dfs[0].iloc[j]["Size"] for j in range(dfs_len)]
    columns = ["Модуль Юнга", "Предел прочности"] + feature_names
    
    return pd.DataFrame(data, columns=columns)


def do(mode):
    dfs = read(mode)
    materials = pd.read_csv(f"source_{mode}.csv", sep=",")
    df = prepare(dfs, materials)
    columns = list(df.columns.difference(["Модуль Юнга", "Предел прочности"]))

    X_train, X_test, y_train, y_test = train_test_split(
        df[columns],
        df[["Модуль Юнга", "Предел прочности"]],
        test_size=0.2,
        random_state=42,
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    net = NeuroNet(len(df.columns) - 2, 2, torch.nn.MSELoss()).fit(
        scaler_X.transform(X_train), scaler_y.transform(y_train)
    )
    y_pred = scaler_y.inverse_transform(net.predict(scaler_X.transform(X_test)))
    print(f'MAE for {mode}s: {mean_absolute_error(y_test.to_numpy(), y_pred)}')


def main():
    do("bridge")
    do("pore")


if __name__ == "__main__":
    main()
