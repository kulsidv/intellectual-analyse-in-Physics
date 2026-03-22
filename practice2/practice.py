import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class NeuroNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y, lr=0.01, epoches=100):
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        losses = []
        self.train()
        for epoch in range(epoches):
            optimizer.zero_grad()
            logits = self(X_tensor)
            loss_value = loss(logits, y_tensor)
            loss_value.backward()
            optimizer.step()
            losses.append(loss_value.item())
        return self, losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self(X_tensor)
            return logits.numpy()


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
    plt.show()


def main():

    df = pd.read_csv("finish_dataset.csv", sep=",")
    df = df.iloc[:, 1:]
    print(df.describe())
    print(df.corr())

    normalizer = MaxAbsScaler()
    normalized_arr = normalizer.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_arr[:, :-1],
        normalized_arr[:, -1],
        test_size=0.2,
        random_state=42
    )

    model, losses = NeuroNet().fit(X_train, y_train, epoches=500)
    y_pred = model.predict(X_test)

    print(f'r2: {r2_score(y_test, y_pred)}')
    print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred)}')
    print(f'mean_squared_error: {mean_squared_error(y_test, y_pred)}')
    plot_loss(losses, save_path='loss_curve.png')


if __name__ == "__main__":
    main()
