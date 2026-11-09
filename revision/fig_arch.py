"""Redrawn CASE architecture diagram (replaces the original Figure 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, BLUE, LIGHT, GRAYF = "#0b0b0b", "#52514e", "#2a78d6", "#e3edfa", "#f2f1ee"
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, text, fc=GRAYF, ec=MUTED, fs=7.2, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else None, linespacing=1.25)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, lw=1, ls=ls))


box(0.5, 17, 13, 12, "Outer\ntraining fold\n$(X_{tr}, y_{tr})$;\nvalidation\nfold $X_{vl}$", fs=6.4)
ax.text(30, 44.5, "Layer 1: base models", ha="center", fontsize=8, fontweight="bold", color=BLUE)
names = ["CatBoost", "Random Forest", "LightGBM", "XGBoost"]
for i, n in enumerate(names):
    y = 34 - i * 7.6
    box(18, y, 24, 5.6, f"{n}\n(own preprocessing pipeline)", fc=LIGHT, ec=BLUE, fs=6.6)
    arrow(14, 23, 18, y + 2.8)
box(18, 2.2, 24, 4.2, "Optional (CASE-H): SVM, LR", fc="white", ec=MUTED, fs=6.4, ls="--")
ax.text(30, 0.2, "3-fold inner CV $\\rightarrow$ OOF probabilities; refit on full fold "
        "$\\rightarrow$ validation probabilities", ha="center", fontsize=5.8, color=MUTED)

box(46.5, 17, 16, 12, "Meta-features\n(OOF, train)\n$M_{tr}: n_{tr}\\times 4K$\n(validation)\n$M_{vl}: n_{vl}\\times 4K$", fs=6.3)
for i in range(4):
    arrow(42.3, 36.8 - i * 7.6, 46.5, 23)
ax.text(72.5, 44.5, "Layer 2", ha="center", fontsize=8, fontweight="bold", color=BLUE)
box(66, 17, 13, 12, "Logistic-\nregression\nmeta-learner\n(multinomial,\n$C=1$)", fc=LIGHT, ec=BLUE, fs=6.4)
arrow(63.3, 23, 66, 23)
ax.text(90, 44.5, "Layer 3", ha="center", fontsize=8, fontweight="bold", color=BLUE)
box(83, 17, 14, 12, "Platt scaling\n(one-vs-rest,\nfitted by 3-fold\nCV on $M_{tr}$)", fc=LIGHT, ec=BLUE, fs=6.4)
arrow(79.8, 23, 83, 23)
box(83, 4.5, 14, 7, "Calibrated class\nprobabilities;\nlabel = argmax", fs=6.4)
arrow(90, 17, 90, 11.8)
fig.savefig("out/figs/fig_case_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("out/figs/fig_case_architecture.pdf", bbox_inches="tight")
