import pandas as pd
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class NeuroNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y, lr=0.01, epoches=100):
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
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
            X_tensor = torch.FloatTensor(X)
            logits = self(X_tensor)
            return logits.numpy()


def vision(df, model):

    sample_ids_to_plot = [1, 6, 22, 9]

    n_samples_total = 28
    temps_actual = [20, 30, 40, 50, 60, 70, 80]
    temps_interp = [25, 35, 45, 55, 65, 75]

    for s_id in sample_ids_to_plot:
        # 1. СБОР ФАКТИЧЕСКИХ ДАННЫХ ДЛЯ ЭТОГО ОБРАЗЦА
        # Так как данные идут блоками, индекс строки = (температура_блок * 28) + (номер_образца - 1)
        actual_mu = []
        base_features_series = None # Запомним признаки одного из блоков для использования

        for i, t in enumerate(temps_actual):
            idx = i * n_samples_total + (s_id - 1)
            row = df.iloc[idx]

            actual_mu.append(row['mu_cP'])

            if base_features_series is None:
                base_features_series = row

        # 2. ПРЕДСКАЗАНИЕ ДЛЯ ПРОМЕЖУТОЧНЫХ ТЕМПЕРАТУР
        pred_mu = []

        for t in temps_interp:
            material = base_features_series.copy()

            material['T'] = t / 80

            X_input = material[1:7].values.reshape(1, -1)

            pred = model.predict(X_input)[0]
            pred_mu.append(pred[1])  # Второй выход - вязкость

        # 3. ПОСТРОЕНИЕ ГРАФИКА
        plt.figure(figsize=(8, 5))

        # Фактические точки (20, 30...80)
        plt.plot(temps_actual, actual_mu, 'o-', label='Фактические (20–80 °C)', color='blue', linewidth=2)

        # Предсказания (25, 35...75)
        plt.plot(temps_interp, pred_mu, 's--', label='Предсказания (25–75 °C)', color='red', linewidth=2)

        plt.xlabel('Температура, °C')
        plt.ylabel('Вязкость (нормализованная)') # Или обычная, если df не нормирован
        plt.title(f'Образец нефти №{s_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_correlation_heatmap(df, figsize=(10, 8), annot=True, cmap='coolwarm'):
    corr = df.corr(method='pearson')

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt='.2f', cmap=cmap, 
                center=0, square=True, linewidths=0.5)
    plt.title('Матрица корреляций Пирсона', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def main():
    df_raw = pd.read_csv("raw_data.csv", sep=",")

    df = pd.DataFrame()
    for t in range(20, 90, 10):
        df_temp = pd.DataFrame(df_raw.iloc[:, 1:7])
        df_temp["T"] = t
        df_temp["mu_cP"] = df_raw[f"mu_cP_{t}C"]
        df = pd.concat([df, df_temp])

    normalizer = MaxAbsScaler()
    normalized_arr = normalizer.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_arr[:, 1:7], normalized_arr[:, [0, -1]], test_size=0.2, random_state=42
    )

    model = NeuroNet().fit(X_train, y_train, epoches=500)
    y_pred = model.predict(X_test)

    print(f"r2: {r2_score(y_test, y_pred)}")
    print(f"mean_absolute_error: {mean_absolute_error(y_test, y_pred)}")
    print(f"mean_squared_error: {mean_squared_error(y_test, y_pred)}")

    df_norm = pd.DataFrame(normalized_arr, columns=df.columns, index=df.index)
    vision(df_norm, model)
    plot_correlation_heatmap(df_norm)


if __name__ == '__main__':
    main()