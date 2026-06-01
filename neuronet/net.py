import torch


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
