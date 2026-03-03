#!/usr/bin/env python3
"""
Output:
  Fig 1a/b  – MSF profiles (GNM / ANM) with ΔMSF
  Fig 1a/b  – MSF zoom (±15 residues around mutation)
  Fig 2a/b  – ΔCross-correlation heatmaps (GNM / ANM)
  Fig 2c    – Per-residue mean |ΔCC| profile
  Fig 3a    – Eigenvector overlap matrices (GNM + ANM side-by-side)
  Fig 3b    – Per-mode diagonal overlap bar chart
  Fig 4     – Hinge-shift eigenvector plot (modes 1–5)
  Fig 5a    – ΔPRS heatmap
  Fig 5b    – Effectiveness & Sensitivity profiles (WT vs MUT)
  Fig 5c    – N-terminal propagation profile
  Composite – 8-panel overview
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ═════════════════════════════════════════════════════════════════════════════
# Style
# ═════════════════════════════════════════════════════════════════════════════

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_WT = "#2166ac"
C_MUT = "#b2182b"
C_DELTA = "#4d4d4d"
C_POS = "#d73027"
C_NEG = "#4575b4"
C_SITE = "#ff7f00"
C_HINGE_WT = "#2166ac"
C_HINGE_MUT = "#b2182b"
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "YlOrRd"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load(base, subdir, fname):
    return np.load(Path(base) / subdir / fname)


def _save(fig, fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1: MSF Difference
# ═════════════════════════════════════════════════════════════════════════════

def fig1_msf(base, fig_dir, mutation_label, mutation_pos, n_modes=20):
    resnums = _load(base, "1_msf_difference", "resnums.npy")
    si = mutation_pos - int(resnums[0])

    for tag, label in [("gnm", "GNM"), ("anm", "ANM")]:
        msf_wt = _load(base, "1_msf_difference", f"{tag}_msf_wt.npy")
        msf_mut = _load(base, "1_msf_difference", f"{tag}_msf_mut.npy")
        delta = _load(base, "1_msf_difference", f"{tag}_delta_msf.npy")

        # Determine top 5 from data
        top5_inc = [int(resnums[i]) for i in np.argsort(delta)[-5:][::-1]]
        top5_dec = [int(resnums[i]) for i in np.argsort(delta)[:5]]

        fig, axes = plt.subplots(2, 1, figsize=(7, 4.8), height_ratios=[2, 1],
                                  sharex=True, gridspec_kw={"hspace": 0.08})

        ax = axes[0]
        ax.plot(resnums, msf_wt, color=C_WT, label="WT", lw=1.0, alpha=0.85)
        ax.plot(resnums, msf_mut, color=C_MUT, label=mutation_label, lw=1.0, alpha=0.85)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.set_ylabel(r"$\langle \delta R_i^2 \rangle$")
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(f"{label} — Mean-Square Fluctuation Profile", fontweight="bold", pad=8)
        ax.tick_params(axis="x", labelbottom=False)

        ax = axes[1]
        ax.fill_between(resnums, delta, 0, where=(delta >= 0), color=C_POS, alpha=0.35, lw=0)
        ax.fill_between(resnums, delta, 0, where=(delta < 0), color=C_NEG, alpha=0.35, lw=0)
        ax.plot(resnums, delta, color=C_DELTA, lw=0.7)
        ax.axhline(0, color="k", lw=0.4, zorder=0)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.set_xlabel("Residue number")
        ax.set_ylabel(r"$\Delta \mathrm{MSF}_i$")

        ax.annotate(f"{mutation_label}",
                     xy=(mutation_pos, delta[si]),
                     xytext=(mutation_pos + 18, delta[si] + (delta.max() - delta.min()) * 0.25),
                     fontsize=7.5, color=C_SITE, fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=C_SITE, lw=0.8))

        r0 = int(resnums[0])
        for res in top5_inc:
            idx = res - r0
            if 0 <= idx < len(delta):
                ax.plot(res, delta[idx], marker="^", ms=5, color=C_POS,
                        zorder=6, markeredgecolor="white", markeredgewidth=0.4)
                ax.annotate(str(res), xy=(res, delta[idx]),
                            xytext=(0, 6), textcoords="offset points",
                            fontsize=6, ha="center", color=C_POS, fontweight="bold")
        for res in top5_dec:
            idx = res - r0
            if 0 <= idx < len(delta):
                ax.plot(res, delta[idx], marker="v", ms=5, color=C_NEG,
                        zorder=6, markeredgecolor="white", markeredgewidth=0.4)
                ax.annotate(str(res), xy=(res, delta[idx]),
                            xytext=(0, -8), textcoords="offset points",
                            fontsize=6, ha="center", va="top", color=C_NEG, fontweight="bold")

        _save(fig, fig_dir, f"fig1_{tag}_msf_difference")

    # ── Zoomed views ──
    ZOOM_HALF = 40
    left = max(int(resnums[0]), mutation_pos - ZOOM_HALF)
    right = min(int(resnums[-1]), mutation_pos + ZOOM_HALF)
    mask = (resnums >= left) & (resnums <= right)
    rn_z = resnums[mask]

    for tag, label in [("gnm", "GNM"), ("anm", "ANM")]:
        msf_wt = _load(base, "1_msf_difference", f"{tag}_msf_wt.npy")[mask]
        msf_mut = _load(base, "1_msf_difference", f"{tag}_msf_mut.npy")[mask]
        delta = _load(base, "1_msf_difference", f"{tag}_delta_msf.npy")[mask]

        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), height_ratios=[1.6, 1],
                                  sharex=True, gridspec_kw={"hspace": 0.08})

        ax = axes[0]
        ax.plot(rn_z, msf_wt, "-", color=C_WT, lw=1.1, label="WT")
        ax.plot(rn_z, msf_mut, "-", color=C_MUT, lw=1.1, label=mutation_label)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.axvspan(mutation_pos - 0.5, mutation_pos + 0.5, color=C_SITE, alpha=0.08, zorder=0)
        ax.set_ylabel(r"$\langle \delta R_i^2 \rangle$")
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(f"{label} — MSF Zoom (res {left}–{right})", fontweight="bold", pad=8)
        ax.tick_params(axis="x", labelbottom=False)

        ax = axes[1]
        colors = [C_POS if d >= 0 else C_NEG for d in delta]
        ax.bar(rn_z, delta, width=1.0, color=colors, alpha=0.6,
               edgecolor=colors, linewidth=0.3)
        ax.axhline(0, color="k", lw=0.4, zorder=0)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.axvspan(mutation_pos - 0.5, mutation_pos + 0.5, color=C_SITE, alpha=0.08, zorder=0)
        ax.set_xlabel("Residue number")
        ax.set_ylabel(r"$\Delta \mathrm{MSF}_i$")

        _save(fig, fig_dir, f"fig1_{tag}_msf_zoom")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2: Cross-Correlation
# ═════════════════════════════════════════════════════════════════════════════

def fig2_crosscorr(base, fig_dir, mutation_label, mutation_pos):
    resnums = _load(base, "2_crosscorr_comparison", "resnums.npy")

    for tag, label in [("gnm", "GNM"), ("anm", "ANM")]:
        delta_cc = _load(base, "2_crosscorr_comparison", f"{tag}_delta_cc.npy")
        vmax = np.percentile(np.abs(delta_cc), 99)
        if vmax == 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(delta_cc, cmap=CMAP_DIV, norm=norm, origin="lower", aspect="equal",
                        extent=[resnums[0], resnums[-1], resnums[0], resnums[-1]])
        ax.axhline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
        ax.axvline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("ΔCC  (MUT − WT)", fontsize=8)
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Residue number")
        ax.set_title(f"{label} — ΔCross-Correlation ({mutation_label})", fontweight="bold", pad=8)
        _save(fig, fig_dir, f"fig2{tag[0]}_delta_crosscorr_{tag}")

    # Per-residue mean |ΔCC|
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True,
                              gridspec_kw={"hspace": 0.12})
    for i, (tag, label) in enumerate([("gnm", "GNM"), ("anm", "ANM")]):
        mean_abs = _load(base, "2_crosscorr_comparison", f"{tag}_mean_abs_delta_cc.npy")
        ax = axes[i]
        ax.fill_between(resnums, mean_abs, 0, color=C_MUT if i else C_WT, alpha=0.25, lw=0)
        ax.plot(resnums, mean_abs, color=C_MUT if i else C_WT, lw=0.9)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.set_ylabel(f"Mean |ΔCC|  ({label})")
        if i == 1:
            ax.set_xlabel("Residue number")
        ax.set_title(f"{label}", fontsize=9, loc="left", fontweight="bold")
    fig.suptitle(f"Per-Residue Mean |ΔCC| ({mutation_label})", fontweight="bold", fontsize=10, y=1.02)
    _save(fig, fig_dir, "fig2c_mean_abs_delta_cc")

    # ΔCC zoom (centred on mutation site)
    ZOOM_HALF_CC = 22
    for tag, label in [("gnm", "GNM"), ("anm", "ANM")]:
        delta_cc = _load(base, "2_crosscorr_comparison", f"{tag}_delta_cc.npy")

        # Find index of mutation residue and build a window around it
        mut_idx = np.argmin(np.abs(resnums - mutation_pos))
        lo = max(0, mut_idx - ZOOM_HALF_CC)
        hi = min(len(resnums), mut_idx + ZOOM_HALF_CC + 1)
        delta_zoom = delta_cc[lo:hi, lo:hi]
        rn_zoom = resnums[lo:hi]

        vmax = np.percentile(np.abs(delta_zoom), 99)
        if vmax == 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(delta_zoom, cmap=CMAP_DIV, norm=norm, origin="lower", aspect="equal",
                        extent=[rn_zoom[0], rn_zoom[-1], rn_zoom[0], rn_zoom[-1]])
        if rn_zoom[0] <= mutation_pos <= rn_zoom[-1]:
            ax.axhline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
            ax.axvline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("ΔCC  (MUT − WT)", fontsize=8)
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Residue number")
        ax.set_title(f"{label} — ΔCC Zoom (res {int(rn_zoom[0])}–{int(rn_zoom[-1])})",
                      fontweight="bold", pad=8)
        _save(fig, fig_dir, f"fig2_{tag}_delta_cc_zoom")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3: Overlap
# ═════════════════════════════════════════════════════════════════════════════

def fig3_overlap(base, fig_dir, mutation_label, n_modes=20):
    # 3a: matrices
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for i, (tag, label) in enumerate([("gnm", "GNM"), ("anm", "ANM")]):
        omat = _load(base, "3_eigenvector_overlap", f"{tag}_overlap_matrix.npy")
        nm = omat.shape[0]
        ax = axes[i]
        im = ax.imshow(omat, cmap="inferno", vmin=0, vmax=1, origin="lower", aspect="equal")
        ax.set_xlabel("MUT mode index")
        ax.set_ylabel("WT mode index")
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(range(0, nm, 2))
        ax.set_xticklabels(range(1, nm + 1, 2), fontsize=7)
        ax.set_yticks(range(0, nm, 2))
        ax.set_yticklabels(range(1, nm + 1, 2), fontsize=7)
        for k in range(nm):
            val = omat[k, k]
            color = "w" if val < 0.85 else "k"
            ax.text(k, k, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03)
    cbar.set_label(r"$|\langle u_i^{\rm WT} | u_j^{\rm MUT} \rangle|$", fontsize=8)
    fig.suptitle(f"Eigenvector Overlap Matrices ({mutation_label})", fontweight="bold", fontsize=10, y=1.02)
    _save(fig, fig_dir, "fig3a_overlap_matrices")

    # 3b: diagonal bar chart
    gnm_diag = _load(base, "3_eigenvector_overlap", "gnm_mode_overlaps.npy")
    anm_diag = _load(base, "3_eigenvector_overlap", "anm_mode_overlaps.npy")
    nm = len(gnm_diag)
    modes = np.arange(1, nm + 1)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.35
    ax.bar(modes - w / 2, gnm_diag, width=w, color=C_WT, alpha=0.85, label="GNM", edgecolor="white", lw=0.3)
    ax.bar(modes + w / 2, anm_diag, width=w, color=C_MUT, alpha=0.85, label="ANM", edgecolor="white", lw=0.3)
    ax.axhline(1.0, color="k", lw=0.4, ls=":", zorder=0)
    ax.axhline(0.95, color="grey", lw=0.3, ls=":", zorder=0, alpha=0.5)
    ax.set_xlabel("Mode index")
    ax.set_ylabel(r"$|\langle u_i^{\rm WT} | u_i^{\rm MUT} \rangle|$")
    ax.set_ylim(max(0, float(min(gnm_diag.min(), anm_diag.min())) - 0.05), 1.02)
    ax.set_xticks(modes)
    ax.legend(frameon=False)
    ax.set_title(f"Per-Mode Diagonal Overlap ({mutation_label})", fontweight="bold", pad=8)

    gnm_rmsip = float(np.load(Path(base) / "3_eigenvector_overlap" / "gnm_rmsip.npy").item())
    anm_rmsip = float(np.load(Path(base) / "3_eigenvector_overlap" / "anm_rmsip.npy").item())
    ax.text(0.98, 0.04,
            f"RMSIP(10):  GNM = {gnm_rmsip:.4f}   ANM = {anm_rmsip:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#333", style="italic",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3", alpha=0.9))
    _save(fig, fig_dir, "fig3b_diagonal_overlap")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4: Hinges
# ═════════════════════════════════════════════════════════════════════════════

def fig4_hinges(base, fig_dir, mutation_label, mutation_pos):
    resnums = _load(base, "4_hinge_shift", "resnums.npy")
    n_modes = 5

    with open(Path(base) / "4_hinge_shift" / "hinge_summary.json") as f:
        hsummary = json.load(f)

    fig, axes = plt.subplots(n_modes, 1, figsize=(7, 8), sharex=True,
                              gridspec_kw={"hspace": 0.25})

    for k in range(n_modes):
        ax = axes[k]
        mode_label = f"mode_{k+1}"
        ev_wt = _load(base, "4_hinge_shift", f"wt_{mode_label}_eigvec.npy")
        ev_mut = _load(base, "4_hinge_shift", f"mut_{mode_label}_eigvec.npy")

        ax.plot(resnums, ev_wt, color=C_WT, lw=1.0, alpha=0.85, label="WT")
        ax.plot(resnums, ev_mut, color=C_MUT, lw=1.0, alpha=0.85, label=mutation_label)
        ax.axhline(0, color="k", lw=0.3, zorder=0)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.6, alpha=0.5, zorder=0)

        info = hsummary["per_mode"][mode_label]
        for h in info["wt_hinges"]:
            ax.plot(h, 0, marker="v", ms=4, color=C_HINGE_WT, zorder=5, clip_on=False)
        for h in info["mut_hinges"]:
            ax.plot(h, 0, marker="^", ms=4, color=C_HINGE_MUT, zorder=5, clip_on=False)
        for h in info.get("gained_in_mutant", []):
            ax.axvline(h, color=C_MUT, lw=0.4, ls=":", alpha=0.5)
        for h in info.get("lost_in_mutant", []):
            ax.axvline(h, color=C_WT, lw=0.4, ls=":", alpha=0.5)

        ax.set_ylabel(f"Mode {k+1}", fontweight="bold", fontsize=8)
        ax.tick_params(axis="y", labelsize=6.5)
        if k == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    axes[-1].set_xlabel("Residue number")
    fig.suptitle(f"GNM Eigenvectors & Hinge Residues ({mutation_label})",
                  fontweight="bold", fontsize=10, y=0.995)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="v", color=C_HINGE_WT, ls="", ms=5, label="WT hinge"),
        Line2D([], [], marker="^", color=C_HINGE_MUT, ls="", ms=5, label="MUT hinge"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left", handletextpad=0.3)
    _save(fig, fig_dir, "fig4_hinge_shift")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 5: PRS
# ═════════════════════════════════════════════════════════════════════════════

def fig5_prs(base, fig_dir, mutation_label, mutation_pos):
    resnums = _load(base, "5_prs_allosteric", "resnums.npy")

    # 5a: ΔPRS heatmap
    delta_prs = _load(base, "5_prs_allosteric", "delta_prs_matrix.npy")
    vmax = np.percentile(np.abs(delta_prs), 99)
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(delta_prs, cmap=CMAP_DIV, norm=norm, origin="lower", aspect="equal",
                    extent=[resnums[0], resnums[-1], resnums[0], resnums[-1]])
    ax.axhline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
    ax.axvline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("ΔPRS  (MUT − WT)", fontsize=8)
    ax.set_xlabel("Responding residue")
    ax.set_ylabel("Perturbed residue")
    ax.set_title(f"ΔPRS Matrix ({mutation_label})", fontweight="bold", pad=8)
    _save(fig, fig_dir, "fig5a_delta_prs_heatmap")

    # 5b: Effectiveness & Sensitivity
    eff_wt = _load(base, "5_prs_allosteric", "wt_effectiveness.npy")
    eff_mut = _load(base, "5_prs_allosteric", "mut_effectiveness.npy")
    sen_wt = _load(base, "5_prs_allosteric", "wt_sensitivity.npy")
    sen_mut = _load(base, "5_prs_allosteric", "mut_sensitivity.npy")

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True,
                              gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    ax.plot(resnums, eff_wt, color=C_WT, lw=0.9, alpha=0.85, label="WT")
    ax.plot(resnums, eff_mut, color=C_MUT, lw=0.9, alpha=0.85, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.7, alpha=0.6, zorder=0)
    ax.set_ylabel("Effectiveness")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("PRS Effectiveness", fontsize=9, loc="left", fontweight="bold")

    ax = axes[1]
    ax.plot(resnums, sen_wt, color=C_WT, lw=0.9, alpha=0.85, label="WT")
    ax.plot(resnums, sen_mut, color=C_MUT, lw=0.9, alpha=0.85, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.7, alpha=0.6, zorder=0)
    ax.set_ylabel("Sensitivity")
    ax.set_xlabel("Residue number")
    ax.set_title("PRS Sensitivity", fontsize=9, loc="left", fontweight="bold")
    fig.suptitle(f"Perturbation Response Scanning ({mutation_label})", fontweight="bold", fontsize=10, y=1.02)
    _save(fig, fig_dir, "fig5b_effectiveness_sensitivity")

    # 5c: N-terminal propagation
    nterm_wt = _load(base, "5_prs_allosteric", "nterm_propagation_wt.npy")
    nterm_mut = _load(base, "5_prs_allosteric", "nterm_propagation_mut.npy")
    delta_nt = _load(base, "5_prs_allosteric", "delta_nterm_propagation.npy")

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), height_ratios=[2, 1],
                              sharex=True, gridspec_kw={"hspace": 0.08})
    ax = axes[0]
    ax.plot(resnums, nterm_wt, color=C_WT, lw=1.0, alpha=0.85, label="WT")
    ax.plot(resnums, nterm_mut, color=C_MUT, lw=1.0, alpha=0.85, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.7, alpha=0.6, zorder=0)
    ax.set_ylabel("PRS response")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(f"Signal Propagation from N-Terminus (res {int(resnums[0])})",
                  fontweight="bold", pad=8)

    ax = axes[1]
    ax.fill_between(resnums, delta_nt, 0, where=(delta_nt >= 0), color=C_POS, alpha=0.35, lw=0)
    ax.fill_between(resnums, delta_nt, 0, where=(delta_nt < 0), color=C_NEG, alpha=0.35, lw=0)
    ax.plot(resnums, delta_nt, color=C_DELTA, lw=0.7)
    ax.axhline(0, color="k", lw=0.4, zorder=0)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.7, alpha=0.6, zorder=0)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("Δ response (MUT − WT)")
    _save(fig, fig_dir, "fig5c_nterm_propagation")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 6: Per-mode variance decomposition
# ═════════════════════════════════════════════════════════════════════════════

def _compute_per_mode_sf(analysis_dir, subdir, is_anm=False):
    """Compute per-mode squared fluctuations from eigenvalues/eigenvectors."""
    d = Path(analysis_dir) / subdir
    ev_path = d / "eigenvalues.npy"
    ec_path = d / "eigenvectors.npy"
    if not (ev_path.exists() and ec_path.exists()):
        # Try pre-computed per_mode_sqflucts.npy
        pm_path = d / "per_mode_sqflucts.npy"
        if pm_path.exists():
            return np.load(pm_path)
        return None
    ev = np.load(ev_path)
    ec = np.load(ec_path)
    n_m = len(ev)
    if is_anm:
        n_r = ec.shape[0] // 3
        per_mode_sf = np.zeros((n_m, n_r))
        for i in range(n_m):
            mode_vec = ec[:, i].reshape(n_r, 3)
            per_mode_sf[i] = (mode_vec**2).sum(axis=1) / ev[i]
    else:
        n_r = ec.shape[0]
        per_mode_sf = np.zeros((n_m, n_r))
        for i in range(n_m):
            per_mode_sf[i] = ec[:, i]**2 / ev[i]
    return per_mode_sf


def fig6_mode_decomposition(analysis_dir, fig_dir, mutation_label, mutation_pos, n_modes=20):
    """Per-mode variance decomposition: global % and site % for GNM & ANM."""
    analysis_dir = Path(analysis_dir)
    fig_dir = Path(fig_dir)

    rn_path = analysis_dir / "gnm_wt" / "resnums.npy"
    if not rn_path.exists():
        rn_path = analysis_dir / "anm_wt" / "resnums.npy"
    if rn_path.exists():
        resnums = np.load(rn_path)
    else:
        print("  warning: no resnums found for mode decomposition, skipping fig6")
        return
    site_idx = int(np.argmin(np.abs(resnums - mutation_pos)))

    datasets = {}
    for tag, subdir_wt, subdir_mut, is_anm in [
        ("GNM", "gnm_wt", "gnm_mut", False),
        ("ANM", "anm_wt", "anm_mut", True),
    ]:
        for state, subdir in [("WT", subdir_wt), ("MUT", subdir_mut)]:
            pm = _compute_per_mode_sf(analysis_dir, subdir, is_anm)
            if pm is None:
                continue
            mode_var = pm.sum(axis=1)
            total_var = mode_var.sum()
            pct_global = 100.0 * mode_var / total_var if total_var > 0 else np.zeros(len(mode_var))
            sf_at_site_total = pm[:, site_idx].sum()
            pct_site = 100.0 * pm[:, site_idx] / sf_at_site_total if sf_at_site_total > 0 else np.zeros(len(mode_var))
            datasets[f"{tag} {state}"] = {
                "pct_global": pct_global,
                "pct_site": pct_site,
                "cum_global": np.cumsum(pct_global),
            }

    if not datasets:
        print("  warning: no per-mode data available, skipping fig6")
        return

    n_show = min(n_modes, min(len(v["pct_global"]) for v in datasets.values()))
    modes = np.arange(1, n_show + 1)

    # ── Figure 6a: Global variance % (stacked area for WT, grouped bars for WT vs MUT) ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle(f"Per-Mode Variance Decomposition — {mutation_label}",
                 fontweight="bold", fontsize=11)

    bar_colors_wt = "#2166ac"
    bar_colors_mut = "#b2182b"
    w = 0.35

    for col, model in enumerate(["GNM", "ANM"]):
        wt_key = f"{model} WT"
        mut_key = f"{model} MUT"
        if wt_key not in datasets or mut_key not in datasets:
            continue

        # Top row: Global variance %
        ax = axes[0, col]
        ax.bar(modes - w/2, datasets[wt_key]["pct_global"][:n_show],
               width=w, color=bar_colors_wt, alpha=0.85, label="WT",
               edgecolor="white", lw=0.3)
        ax.bar(modes + w/2, datasets[mut_key]["pct_global"][:n_show],
               width=w, color=bar_colors_mut, alpha=0.85, label=mutation_label,
               edgecolor="white", lw=0.3)

        # Cumulative line (WT)
        ax2 = ax.twinx()
        ax2.plot(modes, datasets[wt_key]["cum_global"][:n_show],
                 "o-", color="#555555", ms=3, lw=1, alpha=0.7, label="Cum. (WT)")
        ax2.set_ylabel("Cumulative %", fontsize=8, color="#555555")
        ax2.set_ylim(0, 105)
        ax2.tick_params(labelsize=7, colors="#555555")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("#999999")

        ax.set_xlabel("Mode")
        ax.set_ylabel("Global Variance (%)")
        ax.set_xticks(modes)
        ax.set_title(f"{model} — Global Variance per Mode", fontweight="bold", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper right")

        # Bottom row: Site variance %
        ax = axes[1, col]
        ax.bar(modes - w/2, datasets[wt_key]["pct_site"][:n_show],
               width=w, color=bar_colors_wt, alpha=0.85, label="WT",
               edgecolor="white", lw=0.3)
        ax.bar(modes + w/2, datasets[mut_key]["pct_site"][:n_show],
               width=w, color=bar_colors_mut, alpha=0.85, label=mutation_label,
               edgecolor="white", lw=0.3)

        ax.set_xlabel("Mode")
        ax.set_ylabel(f"Variance at Site {mutation_pos} (%)")
        ax.set_xticks(modes)
        ax.set_title(f"{model} — Mode Contribution at Mutation Site",
                     fontweight="bold", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    _save(fig, fig_dir, "fig6_mode_decomposition")

    # ── Figure 6b: Per-mode MSF profiles near mutation site (top 3 dominant modes) ──
    for model, is_anm in [("GNM", False), ("ANM", True)]:
        pm_wt = _compute_per_mode_sf(analysis_dir, f"{model.lower()}_wt", is_anm)
        pm_mut = _compute_per_mode_sf(analysis_dir, f"{model.lower()}_mut", is_anm)
        if pm_wt is None or pm_mut is None:
            continue

        # Find top 3 contributing modes at site (WT)
        site_contribs = pm_wt[:, site_idx]
        top3 = np.argsort(site_contribs)[-3:][::-1]

        ZOOM_HALF = 40
        lo = max(0, site_idx - ZOOM_HALF)
        hi = min(len(resnums), site_idx + ZOOM_HALF + 1)
        rn_zoom = resnums[lo:hi]

        fig, axes_m = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                                    constrained_layout=True)
        fig.suptitle(f"{model} Top-3 Dominant Modes at Site {mutation_pos} — {mutation_label}",
                     fontweight="bold", fontsize=11)

        for row, m_idx in enumerate(top3):
            ax = axes_m[row]
            wt_profile = pm_wt[m_idx, lo:hi]
            mut_profile = pm_mut[m_idx, lo:hi]
            pct_wt = 100.0 * site_contribs[m_idx] / site_contribs.sum()

            ax.fill_between(rn_zoom, wt_profile, alpha=0.2, color=C_WT)
            ax.plot(rn_zoom, wt_profile, color=C_WT, lw=1.2, label="WT")
            ax.fill_between(rn_zoom, mut_profile, alpha=0.2, color=C_MUT)
            ax.plot(rn_zoom, mut_profile, color=C_MUT, lw=1.2, label=mutation_label)
            ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7)

            ax.set_ylabel(f"SqFluct (mode {m_idx+1})")
            ax.set_title(f"Mode {m_idx+1}  ({pct_wt:.1f}% of site variance)",
                         fontsize=9, loc="left")
            ax.legend(frameon=False, fontsize=7, loc="upper right")

        axes_m[-1].set_xlabel("Residue")
        _save(fig, fig_dir, f"fig6b_{model.lower()}_top3_modes_at_site")


# ═════════════════════════════════════════════════════════════════════════════
# Composite
# ═════════════════════════════════════════════════════════════════════════════

def fig_composite(base, fig_dir, mutation_label, mutation_pos, n_modes=20):
    resnums = _load(base, "1_msf_difference", "resnums.npy")

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.35,
                            left=0.08, right=0.95, top=0.95, bottom=0.04)

    # A: GNM ΔMSF
    ax = fig.add_subplot(gs[0, 0])
    delta = _load(base, "1_msf_difference", "gnm_delta_msf.npy")
    ax.fill_between(resnums, delta, 0, where=(delta >= 0), color=C_POS, alpha=0.3, lw=0)
    ax.fill_between(resnums, delta, 0, where=(delta < 0), color=C_NEG, alpha=0.3, lw=0)
    ax.plot(resnums, delta, color=C_DELTA, lw=0.7)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.6, alpha=0.6)
    ax.set_ylabel(r"$\Delta \mathrm{MSF}_i$  (GNM)")
    ax.set_xlabel("Residue")
    ax.set_title("A  GNM MSF Difference", fontweight="bold", loc="left", fontsize=9)

    # B: ANM ΔMSF
    ax = fig.add_subplot(gs[0, 1])
    delta = _load(base, "1_msf_difference", "anm_delta_msf.npy")
    ax.fill_between(resnums, delta, 0, where=(delta >= 0), color=C_POS, alpha=0.3, lw=0)
    ax.fill_between(resnums, delta, 0, where=(delta < 0), color=C_NEG, alpha=0.3, lw=0)
    ax.plot(resnums, delta, color=C_DELTA, lw=0.7)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.6, alpha=0.6)
    ax.set_ylabel(r"$\Delta \mathrm{MSF}_i$  (ANM)")
    ax.set_xlabel("Residue")
    ax.set_title("B  ANM MSF Difference", fontweight="bold", loc="left", fontsize=9)

    # C: ΔCC heatmap (GNM)
    ax = fig.add_subplot(gs[1, 0])
    delta_cc = _load(base, "2_crosscorr_comparison", "gnm_delta_cc.npy")
    vmax = np.percentile(np.abs(delta_cc), 99)
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(delta_cc, cmap=CMAP_DIV, norm=norm, origin="lower", aspect="equal",
                    extent=[resnums[0], resnums[-1], resnums[0], resnums[-1]])
    ax.axhline(mutation_pos, color=C_SITE, lw=0.4, ls="--", alpha=0.5)
    ax.axvline(mutation_pos, color=C_SITE, lw=0.4, ls="--", alpha=0.5)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_title("C  GNM ΔCross-Correlation", fontweight="bold", loc="left", fontsize=9)

    # D: Overlap bar chart
    ax = fig.add_subplot(gs[1, 1])
    gnm_diag = _load(base, "3_eigenvector_overlap", "gnm_mode_overlaps.npy")
    anm_diag = _load(base, "3_eigenvector_overlap", "anm_mode_overlaps.npy")
    nm = len(gnm_diag)
    modes = np.arange(1, nm + 1)
    w = 0.35
    ax.bar(modes - w / 2, gnm_diag, width=w, color=C_WT, alpha=0.85, label="GNM", edgecolor="white", lw=0.3)
    ax.bar(modes + w / 2, anm_diag, width=w, color=C_MUT, alpha=0.85, label="ANM", edgecolor="white", lw=0.3)
    ax.axhline(1.0, color="k", lw=0.3, ls=":")
    ax.set_ylim(max(0, float(min(gnm_diag.min(), anm_diag.min())) - 0.05), 1.02)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Overlap")
    ax.set_xticks(modes[::2])
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("D  Eigenvector Overlap", fontweight="bold", loc="left", fontsize=9)

    # E: Hinges (modes 1, 3, 5)
    gs_e = gs[2, 0].subgridspec(3, 1, hspace=0.15)
    for row, k in enumerate([0, 2, 4]):
        ax = fig.add_subplot(gs_e[row])
        mode_label = f"mode_{k+1}"
        ev_wt = _load(base, "4_hinge_shift", f"wt_{mode_label}_eigvec.npy")
        ev_mut = _load(base, "4_hinge_shift", f"mut_{mode_label}_eigvec.npy")
        rn_h = _load(base, "4_hinge_shift", "resnums.npy")
        ax.plot(rn_h, ev_wt, color=C_WT, lw=0.8, alpha=0.8)
        ax.plot(rn_h, ev_mut, color=C_MUT, lw=0.8, alpha=0.8)
        ax.axhline(0, color="k", lw=0.2)
        ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.5, alpha=0.5)
        ax.set_ylabel(f"m{k+1}", fontsize=7, rotation=0, labelpad=12)
        ax.tick_params(labelsize=6)
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Residue", fontsize=7)
    fig.text(gs[2, 0].get_position(fig).x0 + 0.005,
             gs[2, 0].get_position(fig).y1 + 0.005,
             "E  Hinge Eigenvectors (modes 1, 3, 5)",
             fontweight="bold", fontsize=9, transform=fig.transFigure)

    # F: ΔPRS heatmap
    ax = fig.add_subplot(gs[2, 1])
    delta_prs = _load(base, "5_prs_allosteric", "delta_prs_matrix.npy")
    vmax = np.percentile(np.abs(delta_prs), 99)
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(delta_prs, cmap=CMAP_DIV, norm=norm, origin="lower", aspect="equal",
                    extent=[resnums[0], resnums[-1], resnums[0], resnums[-1]])
    ax.axhline(mutation_pos, color=C_SITE, lw=0.4, ls="--", alpha=0.5)
    ax.axvline(mutation_pos, color=C_SITE, lw=0.4, ls="--", alpha=0.5)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
    ax.set_xlabel("Responding residue")
    ax.set_ylabel("Perturbed residue")
    ax.set_title("F  ΔPRS Matrix", fontweight="bold", loc="left", fontsize=9)

    # G: N-terminal propagation
    ax = fig.add_subplot(gs[3, 0])
    nterm_wt = _load(base, "5_prs_allosteric", "nterm_propagation_wt.npy")
    nterm_mut = _load(base, "5_prs_allosteric", "nterm_propagation_mut.npy")
    rn_p = _load(base, "5_prs_allosteric", "resnums.npy")
    ax.plot(rn_p, nterm_wt, color=C_WT, lw=0.9, alpha=0.85, label="WT")
    ax.plot(rn_p, nterm_mut, color=C_MUT, lw=0.9, alpha=0.85, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.6, alpha=0.6)
    ax.set_xlabel("Residue")
    ax.set_ylabel("PRS response")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("G  N-Term Signal Propagation", fontweight="bold", loc="left", fontsize=9)

    # H: ΔPRS profiles
    ax = fig.add_subplot(gs[3, 1])
    delta_eff = _load(base, "5_prs_allosteric", "delta_effectiveness.npy")
    delta_sen = _load(base, "5_prs_allosteric", "delta_sensitivity.npy")
    ax.plot(rn_p, delta_eff, color=C_WT, lw=0.9, alpha=0.85, label="ΔEffectiveness")
    ax.plot(rn_p, delta_sen, color=C_MUT, lw=0.9, alpha=0.85, label="ΔSensitivity")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.6, alpha=0.6)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Δ PRS metric")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("H  ΔPRS Profiles", fontweight="bold", loc="left", fontsize=9)

    fig.suptitle(f"ENM Pattern Analysis — {mutation_label}",
                  fontweight="bold", fontsize=11, y=0.98)
    _save(fig, fig_dir, "fig_composite_overview")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    data_dir: Path,
    fig_dir: Path,
    mutation_label: str,
    mutation_pos: int,
    n_modes: int = 20,
    analysis_dir: Path = None,
):
    """Generate all publication-grade figures from pre-computed pattern data."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)

    print(f"Generating figures from {data_dir} → {fig_dir}")

    fig1_msf(data_dir, fig_dir, mutation_label, mutation_pos, n_modes)
    fig2_crosscorr(data_dir, fig_dir, mutation_label, mutation_pos)
    fig3_overlap(data_dir, fig_dir, mutation_label, n_modes)
    fig4_hinges(data_dir, fig_dir, mutation_label, mutation_pos)
    fig5_prs(data_dir, fig_dir, mutation_label, mutation_pos)

    # Per-mode decomposition (requires analysis_dir with eigenvectors)
    if analysis_dir is not None:
        fig6_mode_decomposition(analysis_dir, fig_dir, mutation_label, mutation_pos, n_modes)
    else:
        print("  skipping fig6 mode decomposition (no analysis_dir provided)")

    fig_composite(data_dir, fig_dir, mutation_label, mutation_pos, n_modes)

    n = len(list(fig_dir.glob("*.pdf")))
    print(f"\nDone — {n} figures saved (PDF + PNG) in {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate ENM pattern figures")
    parser.add_argument("--datadir", required=True, help="Directory with pattern analysis output")
    parser.add_argument("--label", required=True, help="Mutation label (e.g. V13M)")
    parser.add_argument("--site", type=int, required=True, help="Mutation site (residue number)")
    parser.add_argument("--figdir", default=None, help="Figure output directory (default: datadir/figures)")
    parser.add_argument("--analysis-dir", default=None, help="Analysis directory with eigenvalues/eigenvectors (for mode decomposition)")
    parser.add_argument("--modes", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.datadir)
    fig_dir = Path(args.figdir) if args.figdir else data_dir / "figures"

    generate_all_plots(data_dir, fig_dir, args.label, args.site, args.modes,
                       analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
