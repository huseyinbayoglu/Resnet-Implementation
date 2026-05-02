import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

N = 120
x = np.linspace(-2 * np.pi, 2 * np.pi, N)
y = np.sin(2 * x) * np.cos(0.5 * x) + 0.4 * np.sin(5 * x) #+ 0.05 * np.random.randn(N)

X = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, n_hidden_layers, hidden=32):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, epochs=3000, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
    return loss.item()


shallow = MLP(n_hidden_layers=2, hidden=8)
deep = MLP(n_hidden_layers=8, hidden=64)

loss_s = train(shallow)
loss_d = train(deep)

with torch.no_grad():
    x_dense = torch.linspace(-2 * np.pi, 2 * np.pi, 400).unsqueeze(1)
    y_s = shallow(x_dense).squeeze().numpy()
    y_d = deep(x_dense).squeeze().numpy()

plt.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
fig.suptitle(r"$y = \sin(2x)\cdot\cos(0.5x) + 0.4\sin(5x)$", fontsize=17, y=1.02)

DATA_COLOR = "#1f3a5f"
MODEL_COLOR = "#d6336c"

for ax, y_pred, title in [
    (axes[0], y_s, "2개 레이어"),
    (axes[1], y_d, "8개 레이어"),
]:
    ax.plot(x_dense.squeeze().numpy(), y_pred, color=MODEL_COLOR, linewidth=2.8, label="모델", zorder=1)
    ax.scatter(x, y, color=DATA_COLOR, s=35, label="데이터", zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("x", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.grid(alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("y", fontsize=13)
plt.tight_layout()

plt.savefig(
    "/Users/huseyin/Desktop/Python/AI/Resnet implementation/images/depth_comparison.png",
    dpi=200,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
print(f"Shallow MSE: {loss_s:.4f}  |  Deep MSE: {loss_d:.4f}")
