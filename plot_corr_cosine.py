#!/usr/bin/env python
"""Plot Correlation Cosine vs. ANM Mode Index."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

diag = np.load("results/R2103K/patterns/3_eigenvector_overlap/anm_mode_overlaps.npy")
modes = np.arange(1, len(diag) + 1)

fig, ax = plt.subplots(figsize=(8, 4.5))

markerline, stemlines, baseline = ax.stem(modes, diag, linefmt="-", markerfmt="o", basefmt=" ")
plt.setp(stemlines, linewidth=1.2, color="#2196F3")
plt.setp(markerline, markersize=6, color="#1565C0", zorder=5)

ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.5, label="Perfect (1.0)")
ax.axhline(0.95, color="red", lw=0.8, ls="--", alpha=0.6, label="Threshold (0.95)")

imin = np.argmin(diag)
ax.annotate(
    f"Mode {imin+1}\n{diag[imin]:.4f}",
    xy=(modes[imin], diag[imin]),
    xytext=(modes[imin]+1.5, diag[imin]-0.002),
    fontsize=8, color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1),
    bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.3", alpha=0.9),
)

ax.set_xlabel("ANM Mode Index", fontsize=12)
ax.set_ylabel(
    r"Correlation Cosine  $|\langle \mathbf{u}_i^{\mathrm{WT}} \mid \mathbf{u}_i^{\mathrm{MUT}} \rangle|$",
    fontsize=11,
)
ax.set_title(
    "Correlation Cosine vs. ANM Mode Index  (R2103K)",
    fontweight="bold", fontsize=13, pad=10,
)
ax.set_xticks(modes)
ax.set_xlim(0.5, len(diag) + 0.5)
ax.set_ylim(min(diag.min() - 0.003, 0.993), 1.002)
ax.legend(fontsize=9, loc="lower left")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
outpath = "results/R2103K/figures/correlation_cosine_vs_anm_mode.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight")
fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outpath}")
print(f"Saved: {outpath.replace('.png', '.pdf')}")
