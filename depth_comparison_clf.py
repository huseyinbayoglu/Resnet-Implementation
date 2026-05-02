import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(0)
np.random.seed(0)


def make_spirals(n_per_class=500, noise=0.08):
    n = n_per_class
    theta = np.sqrt(np.random.rand(n)) * 7.0 * np.pi
    r_a = theta + np.pi
    data_a = np.stack([np.cos(theta) * r_a, np.sin(theta) * r_a], axis=1) + noise * np.random.randn(n, 2)
    r_b = -theta - np.pi
    data_b = np.stack([np.cos(theta) * r_b, np.sin(theta) * r_b], axis=1) + noise * np.random.randn(n, 2)
    X = np.vstack([data_a, data_b]) / 10.0
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int64)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


X_np, y_np = make_spirals()
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(y_np, dtype=torch.long)

# Display subset: train on full data, plot fewer points for clarity
n_display = 180
display_idx = np.random.choice(len(X_np), size=n_display * 2, replace=False)
X_disp = X_np[display_idx]
y_disp = y_np[display_idx]


class MLP(nn.Module):
    def __init__(self, n_hidden_layers, hidden=64):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, epochs=15000, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == Y).float().mean().item()
    return acc


shallow = MLP(n_hidden_layers=2, hidden=4)
deep = MLP(n_hidden_layers=8, hidden=64)

acc_s = train(shallow)
acc_d = train(deep)


def decision_grid(model, x_min, x_max, y_min, y_max, steps=300):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(grid), dim=1)[:, 1].numpy().reshape(xx.shape)
    return xx, yy, probs


pad = 0.2
x_min, x_max = X_disp[:, 0].min() - pad, X_disp[:, 0].max() + pad
y_min, y_max = X_disp[:, 1].min() - pad, X_disp[:, 1].max() + pad

xx_s, yy_s, p_s = decision_grid(shallow, x_min, x_max, y_min, y_max)
xx_d, yy_d, p_d = decision_grid(deep, x_min, x_max, y_min, y_max)

plt.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS0 = "#1f3a5f"
CLASS1 = "#d6336c"
bg_cmap = ListedColormap([CLASS0, CLASS1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
fig.suptitle(
    r"$\theta = \sqrt{U(0,1)}\cdot 7\pi,\quad"
    r"(x_1, x_2) = \pm(\theta+\pi)\,(\cos\theta,\,\sin\theta) + \varepsilon$",
    fontsize=15, y=1.02,
)

for ax, (xx, yy, p), title, acc in [
    (axes[0], (xx_s, yy_s, p_s), "2개 레이어", acc_s),
    (axes[1], (xx_d, yy_d, p_d), "8개 레이어", acc_d),
]:
    ax.contourf(xx, yy, p, levels=[0, 0.5, 1], cmap=bg_cmap, alpha=0.15, zorder=0)
    ax.contour(xx, yy, p, levels=[0.5], colors="#444", linewidths=2.0, zorder=1)
    mask0 = y_disp == 0
    ax.scatter(X_disp[mask0, 0], X_disp[mask0, 1], color=CLASS0, s=40,
               zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter(X_disp[~mask0, 0], X_disp[~mask0, 1], color=CLASS1, s=40,
               zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{title}  ({acc*100:.1f}%)", fontsize=16, pad=10)
    ax.set_xlabel(r"$x_1$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

axes[0].set_ylabel(r"$x_2$", fontsize=13)
plt.tight_layout()

plt.savefig(
    "/Users/huseyin/Desktop/Python/AI/Resnet implementation/images/depth_comparison_clf.png",
    dpi=200,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
print(f"Shallow acc: {acc_s*100:.2f}%  |  Deep acc: {acc_d*100:.2f}%")
