import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

        # # Fourier basis: learnable coefficients for sine and cosine terms
        # self.n_freqs = self.n_units // 2  # number of frequencies
        # self.freqs = nn.Parameter(torch.linspace(1, self.n_freqs, self.n_freqs).unsqueeze(0))  # shape (1, n_freqs)
        # self.amp_sin = nn.Linear(input_dim, self.n_freqs, bias=False)
        # self.amp_cos = nn.Linear(input_dim, self.n_freqs, bias=False)

        # class FourierLayer(nn.Module):
        #     def __init__(self, amp_sin, amp_cos, freqs):
        #         super().__init__()
        #         self.amp_sin = amp_sin
        #         self.amp_cos = amp_cos
        #         self.freqs = freqs

        #     def forward(self, x):
        #         # x shape: (batch, input_dim)
        #         # freqs shape: (1, n_freqs)
        #         # Compute dot product for each input with each frequency
        #         # x_proj = x @ self.freqs.T  # (batch, n_freqs)
        #         n_freqs = self.freqs.shape[1]
        #         x_proj = nn.Linear(input_dim, n_freqs)(x)
        #         return self.amp_sin(x) * torch.sin(x_proj) + self.amp_cos(x) * torch.cos(x_proj)

        # self.layers = FourierLayer(self.amp_sin, self.amp_cos, self.freqs)
        # self.out = nn.Linear(self.n_freqs, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))
