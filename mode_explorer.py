#!/usr/bin/env python3
"""
Deep per-mode exploration module.

Identifies the most dynamically relevant normal modes — those dominating
the mutation-site fluctuations AND those governing global variance — then
generates detailed per-mode visualizations:

  • Per-mode MSF profiles (WT vs MUT)
  • Orientational cross-correlation maps (per-mode N×N heatmaps)
  • Porcupine plots / vector-field visualization (ANM only)
  • Structural morphing pseudo-trajectories (multi-model PDB)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_WT = "#2166ac"
C_MUT = "#b2182b"
C_SITE = "#ff7f00"
C_POS = "#d73027"
C_NEG = "#4575b4"
C_DELTA = "#4d4d4d"
CMAP_DIV = "RdBu_r"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ensure(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, d, name):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)


def _load_enm(analysis_dir, model, state):
    """Load eigenvalues, eigenvectors, coords, resnums from analysis_dir/<model>_<state>/."""
    d = Path(analysis_dir) / f"{model}_{state}"
    return {
        "eigenvalues": np.load(d / "eigenvalues.npy"),
        "eigenvectors": np.load(d / "eigenvectors.npy"),
        "coords": np.load(d / "ca_coords.npy"),
        "resnums": np.load(d / "resnums.npy"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════════════════

def rank_modes(eigenvalues, eigenvectors, site_idx, is_anm):
    """Compute per-mode variance contributions and rank by site & global impact."""
    n_modes = len(eigenvalues)

    if is_anm:
        n_res = eigenvectors.shape[0] // 3
        per_mode_sf = np.zeros((n_modes, n_res))
        for k in range(n_modes):
            vec = eigenvectors[:, k].reshape(n_res, 3)
            per_mode_sf[k] = (vec ** 2).sum(axis=1) / eigenvalues[k]
    else:
        n_res = eigenvectors.shape[0]
        per_mode_sf = np.zeros((n_modes, n_res))
        for k in range(n_modes):
            per_mode_sf[k] = eigenvectors[:, k] ** 2 / eigenvalues[k]

    mode_var = per_mode_sf.sum(axis=1)
    total_var = mode_var.sum()
    pct_global = 100.0 * mode_var / total_var if total_var > 0 else np.zeros(n_modes)

    site_sf = per_mode_sf[:, site_idx]
    total_site = site_sf.sum()
    pct_site = 100.0 * site_sf / total_site if total_site > 0 else np.zeros(n_modes)

    return {
        "per_mode_sf": per_mode_sf,
        "mode_var": mode_var,
        "pct_global": pct_global,
        "pct_site": pct_site,
        "top_site": np.argsort(pct_site)[::-1],
        "top_global": np.argsort(pct_global)[::-1],
    }


def compute_orient_cc(eigenvectors, mode_idx, is_anm):
    """Per-mode orientational cross-correlation (N×N).

    ANM: cosine of angle between 3D displacement vectors.
    GNM: covariance contribution  v_i * v_j  (sign = direction).
    """
    if is_anm:
        n_res = eigenvectors.shape[0] // 3
        vec = eigenvectors[:, mode_idx].reshape(n_res, 3)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        normed = vec / norms
        return normed @ normed.T          # cosine similarity, range [-1, +1]
    else:
        v = eigenvectors[:, mode_idx]
        absv = np.abs(v)
        absv = np.where(absv < 1e-12, 1e-12, absv)
        normed = v / absv
        return np.outer(normed, normed)   # sign-product, ±1


# ═════════════════════════════════════════════════════════════════════════════
# Plotting — Mode Ranking
# ═════════════════════════════════════════════════════════════════════════════

def plot_mode_ranking(wt_ranks, mut_ranks, mutation_pos, mutation_label,
                      model_name, out_dir, n_show=None):
    """Grouped bar chart: global variance % and site variance % per mode."""
    pct_g_wt = wt_ranks["pct_global"]
    pct_g_mut = mut_ranks["pct_global"]
    pct_s_wt = wt_ranks["pct_site"]
    pct_s_mut = mut_ranks["pct_site"]
    n_show = n_show or min(len(pct_g_wt), 20)
    modes = np.arange(1, n_show + 1)
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), constrained_layout=True)
    fig.suptitle(f"{model_name} — Mode Rankings ({mutation_label})",
                 fontweight="bold", fontsize=11)

    # Global
    ax = axes[0]
    ax.bar(modes - w / 2, pct_g_wt[:n_show], w, color=C_WT, alpha=0.85,
           label="WT", edgecolor="white", lw=0.3)
    ax.bar(modes + w / 2, pct_g_mut[:n_show], w, color=C_MUT, alpha=0.85,
           label=mutation_label, edgecolor="white", lw=0.3)
    # Cumulative line (WT)
    ax2 = ax.twinx()
    cum = np.cumsum(pct_g_wt[:n_show])
    ax2.plot(modes, cum, "o-", color="#555", ms=3, lw=1, alpha=0.6, label="Cumul. (WT)")
    ax2.set_ylabel("Cumulative %", fontsize=8, color="#555")
    ax2.set_ylim(0, 105)
    ax2.tick_params(labelsize=7, colors="#555")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#999")

    for idx in wt_ranks["top_global"][:3]:
        if idx < n_show:
            ax.annotate("★", (idx + 1, max(pct_g_wt[idx], pct_g_mut[idx])),
                        ha="center", va="bottom", fontsize=9, color=C_SITE)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Global Variance (%)")
    ax.set_xticks(modes)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Global Variance per Mode", fontsize=9, fontweight="bold", loc="left")

    # Site
    ax = axes[1]
    ax.bar(modes - w / 2, pct_s_wt[:n_show], w, color=C_WT, alpha=0.85,
           label="WT", edgecolor="white", lw=0.3)
    ax.bar(modes + w / 2, pct_s_mut[:n_show], w, color=C_MUT, alpha=0.85,
           label=mutation_label, edgecolor="white", lw=0.3)
    for idx in wt_ranks["top_site"][:3]:
        if idx < n_show:
            ax.annotate("★", (idx + 1, max(pct_s_wt[idx], pct_s_mut[idx])),
                        ha="center", va="bottom", fontsize=9, color=C_SITE)
    ax.set_xlabel("Mode")
    ax.set_ylabel(f"Variance at Site {mutation_pos} (%)")
    ax.set_xticks(modes)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Mode Contribution at Mutation Site {mutation_pos}",
                 fontsize=9, fontweight="bold", loc="left")

    _save(fig, out_dir, f"{model_name.lower()}_mode_ranking")


# ═════════════════════════════════════════════════════════════════════════════
# Plotting — Per-Mode MSF
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_mode_msf(sf_wt, sf_mut, mode_idx, resnums, mutation_pos,
                      pct_site_wt, pct_site_mut, eigenval_wt,
                      mutation_label, model_name, out_dir, tag):
    """Full-length MSF + ΔMSF + zoomed view for one mode."""
    delta = sf_mut - sf_wt
    site_idx = int(np.argmin(np.abs(resnums - mutation_pos)))
    ZOOM = 50
    lo = max(0, site_idx - ZOOM)
    hi = min(len(resnums), site_idx + ZOOM + 1)
    rn_z = resnums[lo:hi]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5),
                              height_ratios=[1.5, 0.8, 1.5],
                              constrained_layout=True)
    fig.suptitle(
        f"{model_name} Mode {mode_idx + 1} — Per-Residue Fluctuation ({mutation_label})\n"
        f"[{tag}  |  site WT {pct_site_wt:.1f}%  MUT {pct_site_mut:.1f}%  |  "
        f"λ = {eigenval_wt:.4e}]",
        fontweight="bold", fontsize=10,
    )

    # Panel 1: Full protein MSF
    ax = axes[0]
    ax.fill_between(resnums, sf_wt, alpha=0.15, color=C_WT)
    ax.plot(resnums, sf_wt, color=C_WT, lw=0.9, label="WT")
    ax.fill_between(resnums, sf_mut, alpha=0.15, color=C_MUT)
    ax.plot(resnums, sf_mut, color=C_MUT, lw=0.9, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7)
    ax.set_ylabel("Sq. Fluctuation (mode)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Full Protein", fontsize=9, loc="left")

    # Panel 2: ΔMSF
    ax = axes[1]
    ax.fill_between(resnums, delta, 0, where=(delta >= 0), color=C_POS, alpha=0.3, lw=0)
    ax.fill_between(resnums, delta, 0, where=(delta < 0), color=C_NEG, alpha=0.3, lw=0)
    ax.plot(resnums, delta, color=C_DELTA, lw=0.7)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7)
    ax.set_ylabel("ΔMSF (MUT−WT)")
    ax.set_title("Difference", fontsize=9, loc="left")

    # Panel 3: Zoom
    ax = axes[2]
    ax.fill_between(rn_z, sf_wt[lo:hi], alpha=0.15, color=C_WT)
    ax.plot(rn_z, sf_wt[lo:hi], color=C_WT, lw=1.1, label="WT")
    ax.fill_between(rn_z, sf_mut[lo:hi], alpha=0.15, color=C_MUT)
    ax.plot(rn_z, sf_mut[lo:hi], color=C_MUT, lw=1.1, label=mutation_label)
    ax.axvline(mutation_pos, color=C_SITE, ls="--", lw=0.8, alpha=0.7)
    ax.axvspan(mutation_pos - 0.5, mutation_pos + 0.5, color=C_SITE, alpha=0.08)
    ax.set_ylabel("Sq. Fluctuation (mode)")
    ax.set_xlabel("Residue")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Zoom (res {int(rn_z[0])}–{int(rn_z[-1])})", fontsize=9, loc="left")

    _save(fig, out_dir, f"{model_name.lower()}_mode{mode_idx + 1}_msf_{tag}")


# ═════════════════════════════════════════════════════════════════════════════
# Plotting — Orientational Cross-Correlation Map
# ═════════════════════════════════════════════════════════════════════════════

def plot_orient_cc(cc_wt, cc_mut, mode_idx, resnums, mutation_pos,
                   model_name, mutation_label, out_dir, tag):
    """Three-panel heatmap: WT, MUT, ΔOCC for one mode."""
    delta_cc = cc_mut - cc_wt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(
        f"{model_name} Mode {mode_idx + 1} — Orientational Cross-Correlation "
        f"({mutation_label})  [{tag}]",
        fontweight="bold", fontsize=10,
    )

    extent = [resnums[0], resnums[-1], resnums[0], resnums[-1]]
    titles = ["WT", f"MUT ({mutation_label})", "Δ (MUT − WT)"]
    data_list = [cc_wt, cc_mut, delta_cc]

    for i, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[i]
        if i < 2:
            im = ax.imshow(data, cmap=CMAP_DIV, vmin=-1, vmax=1,
                           origin="lower", aspect="equal", extent=extent)
        else:
            vmax = max(np.percentile(np.abs(data), 99), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap=CMAP_DIV, norm=norm,
                           origin="lower", aspect="equal", extent=extent)
        ax.axhline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.5)
        ax.axvline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")
        ax.set_title(title, fontweight="bold", fontsize=9)

    _save(fig, out_dir,
          f"{model_name.lower()}_mode{mode_idx + 1}_orient_cc_{tag}")

    # ── Zoomed version (±50 residues around mutation site) ──
    mut_idx = int(np.argmin(np.abs(resnums - mutation_pos)))
    ZOOM = 50
    lo = max(0, mut_idx - ZOOM)
    hi = min(len(resnums), mut_idx + ZOOM + 1)
    rn_z = resnums[lo:hi]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(
        f"{model_name} Mode {mode_idx + 1} — Orient. CC Zoom "
        f"(res {int(rn_z[0])}–{int(rn_z[-1])})  [{tag}]",
        fontweight="bold", fontsize=10,
    )
    extent_z = [rn_z[0], rn_z[-1], rn_z[0], rn_z[-1]]
    data_zoom = [cc_wt[lo:hi, lo:hi], cc_mut[lo:hi, lo:hi], delta_cc[lo:hi, lo:hi]]

    for i, (data, title) in enumerate(zip(data_zoom, titles)):
        ax = axes[i]
        if i < 2:
            im = ax.imshow(data, cmap=CMAP_DIV, vmin=-1, vmax=1,
                           origin="lower", aspect="equal", extent=extent_z)
        else:
            vmax = max(np.percentile(np.abs(data), 99), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap=CMAP_DIV, norm=norm,
                           origin="lower", aspect="equal", extent=extent_z)
        if rn_z[0] <= mutation_pos <= rn_z[-1]:
            ax.axhline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.5)
            ax.axvline(mutation_pos, color=C_SITE, lw=0.5, ls="--", alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")
        ax.set_title(title, fontweight="bold", fontsize=9)

    _save(fig, out_dir,
          f"{model_name.lower()}_mode{mode_idx + 1}_orient_cc_zoom_{tag}")


# ═════════════════════════════════════════════════════════════════════════════
# Plotting — Porcupine / Vector-Field (ANM only)
# ═════════════════════════════════════════════════════════════════════════════

def plot_porcupine(coords_wt, coords_mut, disp_wt, disp_mut,
                   mode_idx, resnums, mutation_pos,
                   mutation_label, out_dir, tag):
    """3D porcupine plots for an ANM mode (WT and MUT)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  – registers projection

    site_idx = int(np.argmin(np.abs(resnums - mutation_pos)))

    for state_lbl, coords, disp in [("WT", coords_wt, disp_wt),
                                     (mutation_label, coords_mut, disp_mut)]:
        state_tag = "wt" if state_lbl == "WT" else "mut"
        mag = np.linalg.norm(disp, axis=1)
        max_mag = np.percentile(mag, 98) if mag.max() > 0 else 1.0
        # Scale arrows for visibility (target ~15 Å for top displacement)
        scale = 15.0 / max_mag if max_mag > 0 else 1.0

        # Colour by magnitude
        norm_mag = mag / max_mag if max_mag > 0 else mag

        # ── Full-protein view ──
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Backbone trace
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color="grey", lw=0.3, alpha=0.3)

        # Sub-sample arrows for readability (≤ ~600 arrows)
        step = max(1, len(resnums) // 600)
        indices = list(range(0, len(resnums), step))
        if site_idx not in indices:
            indices.append(site_idx)
        # Also include residues with large displacement
        big = set(np.where(norm_mag > 0.6)[0])
        indices = sorted(set(indices) | big)

        for idx in indices:
            c = plt.cm.YlOrRd(norm_mag[idx])
            ax.quiver(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                      disp[idx, 0] * scale, disp[idx, 1] * scale,
                      disp[idx, 2] * scale,
                      color=c, arrow_length_ratio=0.15, linewidth=0.6)

        # Mutation site sphere
        ax.scatter(*coords[site_idx], color=C_SITE, s=80, zorder=10,
                   edgecolors="black", linewidth=1, label=f"Site {mutation_pos}")

        ax.set_title(f"ANM Mode {mode_idx + 1} Porcupine — {state_lbl}  [{tag}]",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("X (Å)", fontsize=8)
        ax.set_ylabel("Y (Å)", fontsize=8)
        ax.set_zlabel("Z (Å)", fontsize=8)
        ax.legend(fontsize=8)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        _save(fig, out_dir,
              f"anm_mode{mode_idx + 1}_porcupine_{state_tag}_{tag}")

        # ── Zoomed view (25 Å radius around mutation site) ──
        ZOOM_R = 25.0
        dists = np.linalg.norm(coords - coords[site_idx], axis=1)
        zoom_mask = dists < ZOOM_R
        if zoom_mask.sum() > 5:
            zoom_idx = np.where(zoom_mask)[0]
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(coords[zoom_mask, 0], coords[zoom_mask, 1],
                    coords[zoom_mask, 2],
                    color="grey", lw=0.5, alpha=0.4)
            for idx in zoom_idx:
                c = plt.cm.YlOrRd(norm_mag[idx])
                ax.quiver(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                          disp[idx, 0] * scale, disp[idx, 1] * scale,
                          disp[idx, 2] * scale,
                          color=c, arrow_length_ratio=0.15, linewidth=0.8)
            ax.scatter(*coords[site_idx], color=C_SITE, s=100, zorder=10,
                       edgecolors="black", linewidth=1.2,
                       label=f"Site {mutation_pos}")
            ax.set_title(
                f"ANM Mode {mode_idx + 1} Porcupine Zoom ({ZOOM_R:.0f} Å) — "
                f"{state_lbl}  [{tag}]",
                fontweight="bold", fontsize=10,
            )
            ax.set_xlabel("X (Å)", fontsize=8)
            ax.set_ylabel("Y (Å)", fontsize=8)
            ax.set_zlabel("Z (Å)", fontsize=8)
            ax.legend(fontsize=8)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
            _save(fig, out_dir,
                  f"anm_mode{mode_idx + 1}_porcupine_zoom_{state_tag}_{tag}")


# ═════════════════════════════════════════════════════════════════════════════
# Structural Morphing (Pseudo-Trajectory PDB)
# ═════════════════════════════════════════════════════════════════════════════

def write_morph_pdb(coords, disp, resnums, eigenvalue, mode_idx,
                    out_path, n_frames=20, amplitude=3.0, resnames=None):
    """Write a multi-model Cα PDB along a normal-mode trajectory.

    r(t) = r0 + A · sin(2π·t/N) · u / √λ
    A is scaled so max displacement equals *amplitude* Å.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale displacement
    if disp.ndim == 1:
        # GNM: 1-D → displace along z
        disp_3d = np.zeros((len(resnums), 3))
        disp_3d[:, 2] = disp
    else:
        disp_3d = disp.copy()

    disp_scaled = disp_3d / np.sqrt(eigenvalue) if eigenvalue > 0 else disp_3d
    max_d = np.linalg.norm(disp_scaled, axis=1).max()
    if max_d > 0:
        disp_scaled *= amplitude / max_d

    if resnames is None:
        resnames = ["ALA"] * len(resnums)

    with open(out_path, "w") as f:
        f.write(f"REMARK   Mode {mode_idx + 1} pseudo-trajectory "
                f"({n_frames} frames, amplitude {amplitude:.1f} A)\n")
        f.write(f"REMARK   Eigenvalue: {eigenvalue:.6e}\n")
        for frame in range(n_frames):
            t = np.sin(2.0 * np.pi * frame / n_frames)
            f.write(f"MODEL     {frame + 1:4d}\n")
            for i, resnum in enumerate(resnums):
                xyz = coords[i] + t * disp_scaled[i]
                rn = resnames[i] if i < len(resnames) else "ALA"
                bfact = abs(t) * np.linalg.norm(disp_scaled[i])
                f.write(
                    f"ATOM  {i + 1:5d}  CA  {rn:>3s} A{int(resnum):4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00{bfact:6.2f}           C  \n"
                )
            f.write("ENDMDL\n")
        f.write("END\n")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_mode_exploration(
    wt_pdb,
    mut_pdb,
    analysis_dir,
    out_dir,
    mutation_label: str,
    mutation_pos: int,
    n_modes: int = 20,
    n_top: int = 5,
    morph_frames: int = 20,
    morph_amplitude: float = 3.0,
) -> dict:
    """Deep per-mode exploration for GNM and ANM (separately)."""

    import prody

    print("=" * 70)
    print(f"Mode Exploration: {mutation_label}")
    print("=" * 70)

    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir)
    _ensure(out_dir)

    # Get residue names from PDB
    struct_wt = prody.parsePDB(str(wt_pdb))
    ca_wt_atoms = struct_wt.select("calpha")
    resnames_wt = list(ca_wt_atoms.getResnames())

    struct_mut = prody.parsePDB(str(mut_pdb))
    ca_mut_atoms = struct_mut.select("calpha")
    resnames_mut = list(ca_mut_atoms.getResnames())

    summary = {"mutation": mutation_label, "mutation_pos": mutation_pos}

    for model, is_anm in [("gnm", False), ("anm", True)]:
        MODEL = model.upper()
        print(f"\n{'─' * 50}")
        print(f"  {MODEL} Mode Exploration")
        print(f"{'─' * 50}")

        model_dir = _ensure(out_dir / model)

        # Load data
        try:
            wt_data = _load_enm(analysis_dir, model, "wt")
            mut_data = _load_enm(analysis_dir, model, "mut")
        except FileNotFoundError as e:
            print(f"  Skipping {MODEL}: {e}")
            continue

        resnums = wt_data["resnums"]
        site_idx = int(np.argmin(np.abs(resnums - mutation_pos)))

        # Rank modes
        wt_ranks = rank_modes(wt_data["eigenvalues"], wt_data["eigenvectors"],
                              site_idx, is_anm)
        mut_ranks = rank_modes(mut_data["eigenvalues"], mut_data["eigenvectors"],
                               site_idx, is_anm)

        top_site = wt_ranks["top_site"][:n_top]
        top_global = wt_ranks["top_global"][:n_top]
        all_modes = sorted(set(top_site.tolist()) | set(top_global.tolist()))

        print(f"  Top {n_top} modes at site {mutation_pos}: "
              f"{[m + 1 for m in top_site]}")
        print(f"  Top {n_top} global modes: "
              f"{[m + 1 for m in top_global]}")
        print(f"  Exploring {len(all_modes)} unique modes: "
              f"{[m + 1 for m in all_modes]}")

        # ── Mode ranking plot ──
        plot_mode_ranking(wt_ranks, mut_ranks, mutation_pos,
                          mutation_label, MODEL, model_dir)
        print(f"  ✓ Mode ranking plot saved")

        # ── Per-mode deep analysis ──
        for mode_idx in all_modes:
            m_num = mode_idx + 1
            cats = []
            if mode_idx in top_site:
                cats.append("site")
            if mode_idx in top_global:
                cats.append("global")
            tag = "+".join(cats)

            print(f"\n  Mode {m_num} ({tag}):")

            # Per-mode MSF
            sf_wt = wt_ranks["per_mode_sf"][mode_idx]
            sf_mut = mut_ranks["per_mode_sf"][mode_idx]
            plot_per_mode_msf(
                sf_wt, sf_mut, mode_idx, resnums, mutation_pos,
                float(wt_ranks["pct_site"][mode_idx]),
                float(mut_ranks["pct_site"][mode_idx]),
                float(wt_data["eigenvalues"][mode_idx]),
                mutation_label, MODEL, model_dir, tag,
            )
            print(f"    ✓ MSF profile")

            # Orientational cross-correlation
            cc_wt = compute_orient_cc(wt_data["eigenvectors"], mode_idx, is_anm)
            cc_mut = compute_orient_cc(mut_data["eigenvectors"], mode_idx, is_anm)
            # Align sign for meaningful delta
            v_wt = wt_data["eigenvectors"][:, mode_idx]
            v_mut = mut_data["eigenvectors"][:, mode_idx]
            if np.dot(v_wt, v_mut) < 0:
                # Flipping all signs → CC = outer(−v, −v) = outer(v,v), no change
                pass

            plot_orient_cc(
                cc_wt, cc_mut, mode_idx, resnums, mutation_pos,
                MODEL, mutation_label, model_dir, tag,
            )
            print(f"    ✓ Orientational CC maps (full + zoom)")

            # Porcupine (ANM only)
            if is_anm:
                n_res = wt_data["eigenvectors"].shape[0] // 3
                disp_wt = wt_data["eigenvectors"][:, mode_idx].reshape(n_res, 3)
                disp_mut = mut_data["eigenvectors"][:, mode_idx].reshape(n_res, 3)
                if np.dot(wt_data["eigenvectors"][:, mode_idx],
                          mut_data["eigenvectors"][:, mode_idx]) < 0:
                    disp_mut = -disp_mut

                try:
                    plot_porcupine(
                        wt_data["coords"], mut_data["coords"],
                        disp_wt, disp_mut,
                        mode_idx, resnums, mutation_pos,
                        mutation_label, model_dir, tag,
                    )
                    print(f"    ✓ Porcupine plots (full + zoom, WT + MUT)")
                except Exception as exc:
                    print(f"    ⚠ Porcupine plot failed: {exc}")

                # Structural morphing PDBs
                write_morph_pdb(
                    wt_data["coords"], disp_wt, resnums,
                    float(wt_data["eigenvalues"][mode_idx]), mode_idx,
                    model_dir / f"mode{m_num}_morph_wt_{tag}.pdb",
                    morph_frames, morph_amplitude, resnames_wt,
                )
                write_morph_pdb(
                    mut_data["coords"], disp_mut, mut_data["resnums"],
                    float(mut_data["eigenvalues"][mode_idx]), mode_idx,
                    model_dir / f"mode{m_num}_morph_mut_{tag}.pdb",
                    morph_frames, morph_amplitude, resnames_mut,
                )
                print(f"    ✓ Morph PDBs")

            else:
                # GNM morph (z-displacement)
                disp_wt = wt_data["eigenvectors"][:, mode_idx]
                disp_mut = mut_data["eigenvectors"][:, mode_idx]
                if np.dot(disp_wt, disp_mut) < 0:
                    disp_mut = -disp_mut
                write_morph_pdb(
                    wt_data["coords"], disp_wt, resnums,
                    float(wt_data["eigenvalues"][mode_idx]), mode_idx,
                    model_dir / f"mode{m_num}_morph_wt_{tag}.pdb",
                    morph_frames, morph_amplitude, resnames_wt,
                )
                write_morph_pdb(
                    mut_data["coords"], disp_mut, mut_data["resnums"],
                    float(mut_data["eigenvalues"][mode_idx]), mode_idx,
                    model_dir / f"mode{m_num}_morph_mut_{tag}.pdb",
                    morph_frames, morph_amplitude, resnames_mut,
                )
                print(f"    ✓ Morph PDBs")

        # Save model summary
        model_summary = {
            "n_modes_analyzed": len(all_modes),
            "modes_analyzed": [int(m + 1) for m in all_modes],
            "top_site_modes": [int(m + 1) for m in top_site],
            "top_global_modes": [int(m + 1) for m in top_global],
            "pct_global_wt": wt_ranks["pct_global"].tolist(),
            "pct_global_mut": mut_ranks["pct_global"].tolist(),
            "pct_site_wt": wt_ranks["pct_site"].tolist(),
            "pct_site_mut": mut_ranks["pct_site"].tolist(),
        }
        summary[model] = model_summary

    with open(out_dir / "exploration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_figs = len(list(out_dir.rglob("*.pdf")))
    n_pdbs = len(list(out_dir.rglob("*.pdb")))
    print(f"\n{'=' * 70}")
    print(f"Mode exploration complete: {n_figs} figures, {n_pdbs} morph PDBs")
    print(f"All outputs → {out_dir}/")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Deep per-mode exploration of normal-mode dynamics",
    )
    parser.add_argument("--wt", required=True, help="Wild-type PDB file")
    parser.add_argument("--mut", required=True, help="Mutant PDB file")
    parser.add_argument("--analysis-dir", required=True,
                        help="Analysis directory (with gnm_wt/, anm_wt/ etc.)")
    parser.add_argument("--outdir", default="mode_explorer",
                        help="Output directory")
    parser.add_argument("--label", required=True, help="Mutation label (e.g. R2103K)")
    parser.add_argument("--site", type=int, required=True, help="Mutation position")
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--n-top", type=int, default=5,
                        help="Number of top modes to explore (default: 5)")
    parser.add_argument("--morph-frames", type=int, default=20)
    parser.add_argument("--morph-amplitude", type=float, default=3.0,
                        help="Max morph displacement in Å (default: 3.0)")
    args = parser.parse_args()

    run_mode_exploration(
        wt_pdb=Path(args.wt),
        mut_pdb=Path(args.mut),
        analysis_dir=Path(args.analysis_dir),
        out_dir=Path(args.outdir),
        mutation_label=args.label,
        mutation_pos=args.site,
        n_modes=args.modes,
        n_top=args.n_top,
        morph_frames=args.morph_frames,
        morph_amplitude=args.morph_amplitude,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
