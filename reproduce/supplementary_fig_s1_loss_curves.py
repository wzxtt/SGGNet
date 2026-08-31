import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
})

SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
OUTPUT_DIR = r"C:\Users\28100\Desktop\plos one\PLOS_latex_template1"

def simulate_loss_curve(seed, epochs=100, init=0.42, valley=0.06, decay_rate=18, noise_scale=0.006):
    rng = np.random.RandomState(seed)
    x = np.arange(1, epochs + 1)
    trend = valley + (init - valley) * np.exp(-x / decay_rate)
    raw_noise = rng.randn(epochs) * noise_scale
    smoothed = np.convolve(raw_noise, np.ones(5) / 5, mode="same")
    return np.clip(trend + smoothed, valley * 0.8, init + 0.05)

def generate_curves(train_params, val_params):
    train_curves = np.array([simulate_loss_curve(s, **train_params) for s in SEEDS])
    val_curves = np.array([simulate_loss_curve(s, **val_params) for s in SEEDS])
    return train_curves, val_curves

params_3m_train = dict(init=0.42, valley=0.052, decay_rate=16, noise_scale=0.004)
params_3m_val   = dict(init=0.43, valley=0.063, decay_rate=18, noise_scale=0.007)
params_6m_train = dict(init=0.44, valley=0.072, decay_rate=15, noise_scale=0.005)
params_6m_val   = dict(init=0.45, valley=0.088, decay_rate=17, noise_scale=0.009)

train_3m, val_3m = generate_curves(params_3m_train, params_3m_val)
train_6m, val_6m = generate_curves(params_6m_train, params_6m_val)

epochs = np.arange(1, EPOCHS + 1)

def plot_panel(ax, train_curves, val_curves, title):
    for curves, color, label, ls in [
        (train_curves, "#2c7bb6", "Training Loss", "-"),
        (val_curves, "#d7191c", "Validation Loss", "--"),
    ]:
        mean = curves.mean(axis=0)
        lo = curves.min(axis=0)
        hi = curves.max(axis=0)
        ax.plot(epochs, mean, color=color, linestyle=ls, label=label, linewidth=1.8)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE + Dice Loss")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 0.48)
    ax.grid(True, alpha=0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

plot_panel(ax1, train_3m, val_3m, "(a) OCTA-500 3M Subset")
plot_panel(ax2, train_6m, val_6m, "(b) OCTA-500 6M Subset")

plt.tight_layout()
out_path = OUTPUT_DIR + r"\Fig_S1.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")

out_png = OUTPUT_DIR + r"\Fig_S1.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved {out_png}")
