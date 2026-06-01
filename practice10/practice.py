import re
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MaxAbsScaler


class NeuroNet(torch.nn.Module):
    def __init__(self, input_nodes, output_nodes, loss_func):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_nodes, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, output_nodes),
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
            return torch.sigmoid(logits).numpy()


def parsing(formula: str):
    pattern = r'[A-Z]{1}[a-z]{0,1}'
    elems = re.findall(pattern, formula)
    return elems


def main():
    df = pd.read_excel('data_piezo.xlsx', sheet_name=0)
    
    elems = set()
    df.loc[:, 'formula'].map(lambda formula: elems.update(parsing(formula)))
    elems = sorted(list(elems))

    for elem in elems:
        df[elem] = df['formula'].str.contains(elem)

    
    df.replace({True: 1, False: 0}, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, ['space group', 'piezoelectric modulus (C/m^2)']],
        df.loc[:, elems],
        test_size=0.2,
        random_state=42
    )

    cat_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', 
        unknown_value=-1
    )
    X_train[['space group']] = cat_encoder.fit_transform(X_train[['space group']])
    X_test[['space group']]  = cat_encoder.transform(X_test[['space group']])

    num_scaler = MaxAbsScaler()
    X_train[['piezoelectric modulus (C/m^2)']] = num_scaler.fit_transform(
        X_train[['piezoelectric modulus (C/m^2)']]
    )
    X_test[['piezoelectric modulus (C/m^2)']]  = num_scaler.transform(
        X_test[['piezoelectric modulus (C/m^2)']]
    )
    print(y_test.iloc[:3].to_string())

    module_scaler = MaxAbsScaler()
    X_train[['piezoelectric modulus (C/m^2)']] = module_scaler.fit_transform(X_train[['piezoelectric modulus (C/m^2)']])
    X_test[['piezoelectric modulus (C/m^2)']]  = module_scaler.transform(X_test[['piezoelectric modulus (C/m^2)']])

    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')
    net = NeuroNet(2, len(elems), torch.nn.BCEWithLogitsLoss())
    net.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = net.predict(X_test.to_numpy())
    y_pred_rounded = np.round(y_pred, 3)
    print(y_pred_rounded[:3])

    

if __name__ == '__main__':
    main()