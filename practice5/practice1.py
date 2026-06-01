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
    rows = []
    for i in range(1, count + 1):
        title = f"{mode+'s'}/Dist_{mode}_{i}.txt"
        with open(title, "r", encoding="UTF-8") as file:
            lines = file.readlines()
            row = dict()
            for j in range(0, 15, 2):
                key = lines[j].strip()
                value = lines[j + 1].strip().replace(",", ".")
                row[key] = float(value)
            key = lines[16].strip()
            row[key] = [
                list(map(lambda x: float(x.replace(",", ".")), line.split()))
                for line in lines[17:]
            ]
            rows.append(row)
    return pd.DataFrame(rows)


def params(df):
    df["Средний размер"] = df["Распределение"].apply(
        lambda x: np.mean([items[0] for items in x])
    )
    df["Максимальное количество"] = df["Распределение"].apply(
        lambda x: max(x, key=lambda items: items[1])[1]
    )
    df["Размер, соответствующий максимальному количеству"] = df["Распределение"].apply(
        lambda x: max(x, key=lambda items: items[1])[0]
    )
    df["Диапазон изменения размеров"] = df["Распределение"].apply(
        lambda x: max(x, key=lambda items: items[0])[0]
        - min(x, key=lambda items: items[0])[0]
    )
    df["S1"] = df["Распределение"].apply(lambda x: sum([items[1] for items in x]))
    df["S2"] = df["Распределение"].apply(
        lambda x: sum([items[1] * items[0] for items in x])
    )
    df["S3"] = df["Распределение"].apply(
        lambda x: sum([items[1] * items[0] * items[0] for items in x])
    )


def do(mode):
    df = read(mode)
    params(df)
    df.drop("Распределение", axis=1, inplace=True)
    columns = [
        "Средний размер",
        "Максимальное количество",
        "Размер, соответствующий максимальному количеству",
        "Диапазон изменения размеров",
        "S1",
        "S2",
        "S3",
        "Пористость",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[columns],
        df[["Модуль Юнга", "Предел прочности"]],
        test_size=0.2,
        random_state=42,
    )

    scaler_X = StandardScaler().fit(df[columns])
    scaler_y = StandardScaler().fit(df[["Модуль Юнга", "Предел прочности"]])

    X_train, y_train = scaler_X.transform(X_train), scaler_y.transform(y_train)
    model = NeuroNet(len(columns), 2, torch.nn.MSELoss()).fit(X_train, y_train, epoches=350)
    y_pred = model.predict(scaler_X.transform(X_test))
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    print(f"mean_absolute_error: {mean_absolute_error(y_test.to_numpy(), y_pred_orig)}")
    print(f"mean_squared_error: {mean_squared_error(y_test.to_numpy(), y_pred_orig)}")


def main():
    do("bridge")
    do("pore")


if __name__ == "__main__":
    main()
