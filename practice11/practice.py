import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


TEMPS = [1823, 1900, 2000, 2100, 2200, 2300, 2400]
WAVE_NUMS = [0.083, 0.166, 0.249]
spectrum_prefix = "Spectrum/Results_T"
frequency_prefix = "Frequency/ResultsCT_T"


class NeuroNet(torch.nn.Module):
    def __init__(self, input_nodes, output_nodes, loss_func):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_nodes, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, output_nodes),
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


def main():
    spectrum_dfs_by_wave_nums = []

    # считываем все частоты с волновым числом 0.83, 0.166, 0.249
    # разделяя их по волновому числу, чтобы по нему можно были сделать join,
    # указав только температуру
    for wave_num in WAVE_NUMS:
        all_temps_for_this_wave_num = pd.DataFrame(columns=['Frequency', 'Amplitude', 'Temp'])
        for temp in TEMPS:
            df = pd.read_csv(
                f"{spectrum_prefix}{temp}K/CT_k{str(wave_num).replace('.', ',')}.txt",
                sep=' ',
                header=None,
                names=['Frequency', 'Amplitude']
            )
            df['Temp'] = temp
            all_temps_for_this_wave_num = pd.concat([all_temps_for_this_wave_num, df])
        spectrum_dfs_by_wave_nums.append(all_temps_for_this_wave_num)

    spectrum_083 = spectrum_dfs_by_wave_nums[0]
    spectrum_166 = spectrum_dfs_by_wave_nums[1]
    spectrum_249 = spectrum_dfs_by_wave_nums[2]

    rows_083 = []
    rows_166 = []
    rows_249 = []
    for temp in TEMPS:
        row_083 = {'Temp': temp}
        row_166 = {'Temp': temp}
        row_249 = {'Temp': temp}
        for i in range(1, 5):
            delta_i = pd.read_csv(
                f"{frequency_prefix}{temp}K/Delta{i}.txt",
                sep=' ',
                header=None,
                names=['wave_num', f'Delta {i}'],
                nrows=3
            )
            row_083[f'Delta {i}'] = delta_i.loc[0, f'Delta {i}']
            row_166[f'Delta {i}'] = delta_i.loc[1, f'Delta {i}']
            row_249[f'Delta {i}'] = delta_i.loc[2, f'Delta {i}']
        rows_083.append(row_083)
        rows_166.append(row_166)
        rows_249.append(row_249)

    frequency_083 = pd.DataFrame(rows_083, index=None)
    frequency_166 = pd.DataFrame(rows_166, index=None)
    frequency_249 = pd.DataFrame(rows_249, index=None)

    result_083 = spectrum_083.merge(frequency_083, on='Temp')
    result_166 = spectrum_166.merge(frequency_166, on='Temp')
    result_249 = spectrum_249.merge(frequency_249, on='Temp')

    result_083['Wave num'] = 0.083
    result_166['Wave num'] = 0.166
    result_249['Wave num'] = 0.249

    result_df = pd.concat([result_083, result_166, result_249])

    X_train, X_test, y_train, y_test = train_test_split(
        result_df[['Amplitude', 'Wave num', 'Temp']],
        result_df[[f'Delta {i}' for i in range(1, 5)]],
        test_size=0.2,
        random_state=42
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    model = NeuroNet(3, 4, torch.nn.MSELoss())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_perd = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
    print(x_scaler.inverse_transform(X_test)[:5])
    print(y_perd[:5])
    print(mean_absolute_error(y_test, y_perd))

    y_pred_another = model.predict(x_scaler.transform(np.array([[4.19276186907499, 0.117, 1823]])))
    print(y_scaler.inverse_transform(y_pred_another))


if __name__ == "__main__":
    main()
