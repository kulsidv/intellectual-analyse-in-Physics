import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


ATOM_COUNT = 256


class NeuroNet(torch.nn.Module):
    def __init__(self, input_nodes, output_nodes, loss_func):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_nodes, 800),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(800, 550),
            torch.nn.SiLU(),
            torch.nn.Linear(550, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, output_nodes)
        )
        self.loss_func = loss_func

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y, lr=0.01, epochs=100, l1_lambda=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X.to_numpy())
        # CrossEntropyLoss требует целые метки классов (0, 1, 2) ← ОСТАВЛЕНО КАК БЫЛО
        y_tensor = torch.LongTensor(y.to_numpy().ravel())

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self(X_tensor)
            
            # 1. Основная функция потерь (CrossEntropyLoss)
            task_loss = self.loss_func(logits, y_tensor)
            
            # 2. L1-штраф: сумма модулей всех весов матриц (bias не штрафуем)
            l1_penalty = sum(torch.abs(p).sum() for p in self.parameters() if p.dim() > 1)
            
            # 3. Итоговый лосс = задача + регуляризация
            loss = task_loss + l1_lambda * l1_penalty
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} | "
                      f"Task: {task_loss.item():.4f} | L1: {(l1_lambda * l1_penalty).item():.4f}")
        return self

    def predict(self, X, return_probs=True):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.to_numpy())
            logits = self(X_tensor)
            if return_probs:
                # Softmax применяется ТОЛЬКО здесь для интерпретации
                return torch.softmax(logits, dim=1)
            return logits


def read_file(title):

    if "cryst" in title:
        state = 0
    elif "liquid" in title:
        state = 1
    else:
        state = 2

    rows = []
    with open(title, "r", encoding="UTF-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line.strip() == "ITEM: BOX BOUNDS":
                x_line = list(map(float, file.readline().split()))
                x_box = x_line[1] - x_line[0]
                y_line = list(map(float, file.readline().split()))
                y_box = y_line[1] - y_line[0]
                z_line = list(map(float, file.readline().split()))
                z_box = z_line[1] - z_line[0]
            elif line.strip() == "ITEM: ATOMS id type x y z":
                row = dict()
                for _ in range(ATOM_COUNT):
                    id, type, x, y, z = file.readline().split()
                    row[f"{id}_x"] = float(x) / x_box
                    row[f"{id}_y"] = float(y) / y_box
                    row[f"{id}_z"] = float(z) / z_box
                row["state"] = state
                rows.append(row)
    return pd.DataFrame(rows)


def get_data(file_name: str, atom_count: int) -> pd.DataFrame:
    rows = []
    with open(file_name, "r", encoding="UTF-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line.strip() == "ITEM: TIMESTEP":
                timestep = int(file.readline())
            elif line.strip() == "ITEM: BOX BOUNDS":
                x_line = list(map(float, file.readline().split()))
                x_box = x_line[1] - x_line[0]
                y_line = list(map(float, file.readline().split()))
                y_box = y_line[1] - y_line[0]
                z_line = list(map(float, file.readline().split()))
                z_box = z_line[1] - z_line[0]
            elif line.strip() == "ITEM: ATOMS id type x y z":
                row = dict()
                for _ in range(atom_count):
                    id, type, x, y, z = file.readline().split()
                    row[f"{id}_x"] = float(x) / x_box
                    row[f"{id}_y"] = float(y) / y_box
                    row[f"{id}_z"] = float(z) / z_box
                row["timestep"] = timestep
                rows.append(row)
    return pd.DataFrame(rows)


def predict_and_fix(data: pd.DataFrame, model: NeuroNet, input_cols: list[str]) -> pd.DataFrame:
    predictions = model.predict(data[input_cols])
    result = pd.DataFrame(predictions.numpy(), columns=["cryst", "liquid", "gas"])
    result["timestep"] = data["timestep"].values
    return result


def main():
    cryst = read_file('dump_cryst_N256_stp100.txt')
    liquid = read_file('dump_liquid_N256_stp100.txt')
    gas = read_file('dump_gas_N256_stp100.txt')

    df = pd.concat([cryst, liquid, gas], axis=0)

    input_cols = list(df.columns)
    input_cols.remove('state')

    X_train, X_test, y_train, y_test = train_test_split(
        df[input_cols],
        df['state'],
        test_size=0.2,
        random_state=42
    )

    model = NeuroNet(
        input_nodes=ATOM_COUNT * 3,
        output_nodes=3,
        loss_func=torch.nn.CrossEntropyLoss()
    ).fit(X_train, y_train, lr=0.001, epochs=50)

    probs = model.predict(X_test)
    y_pred_classes = torch.argmax(probs, dim=1).numpy()

    acc = accuracy_score(y_test, y_pred_classes)
    print(f"\nТочность на тесте: {acc:.2%}")

    df1 = get_data("sample5_melt_N256_stp1000_press20.0.txt", ATOM_COUNT)
    result = predict_and_fix(df1, model, input_cols)

    plt.plot(result['timestep'] * 0.00044, result['cryst'], label='cryst', color='blue')
    plt.plot(result['timestep'] * 0.00044, result['liquid'], label='liquid', color='green')
    plt.plot(result['timestep'] * 0.00044, result['gas'], label='gas', color='red')
    plt.show()


if __name__ == "__main__":
    main()