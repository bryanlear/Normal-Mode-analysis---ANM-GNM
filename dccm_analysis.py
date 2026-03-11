#!/usr/bin/env python3
"""
Dynamic Cross-Correlation Map (DCCM) analysis — generic pipeline module.

Computes GNM and ANM cross-correlation matrices from eigenvalues/eigenvectors,
properly excluding ANM rigid-body modes, then generates:
  - Cross-correlation matrices for WT and MUT
  - Difference maps (ΔC = C_mut - C_wt)
  - Coupling-to-mutation profiles
  - Quantitative summaries
  - Publication-quality heatmaps and profiles
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from enm_analysis import detect_rigid_body_modes


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
# Correlation computation (from raw eigenvalues/eigenvectors)
# ═════════════════════════════════════════════════════════════════════════════

def compute_gnm_corr(eigenvalues, eigenvectors):
    """Compute GNM cross-correlation from internal modes.

    GNM eigenvectors: (N, n_modes), scalar per residue per mode.
    Cov_ij = sum_k (u_ik * u_jk / lambda_k)
    C_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    """
    inv_eig = 1.0 / eigenvalues
    W = eigenvectors * np.sqrt(inv_eig)[np.newaxis, :]
    cov = W @ W.T
    diag = np.sqrt(np.diag(cov))
    diag[diag == 0] = 1e-30
    corr = cov / np.outer(diag, diag)
    np.clip(corr, -1, 1, out=corr)
    return corr, cov


def compute_anm_corr(eigenvalues, eigenvectors, n_rigid=5):
    """Compute ANM cross-correlation from internal modes (rigid-body excluded).

    ANM eigenvectors: (3N, n_modes). Reshape per residue: (N, 3, n_modes).
    Cov_ij = sum_k (1/lambda_k) * (u_i,k . u_j,k)  [dot product over xyz]
    C_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    """
    n_total = eigenvectors.shape[0]
    N = n_total // 3

    int_eig = eigenvalues[n_rigid:]
    int_evec = eigenvectors[:, n_rigid:]
    n_int = int_evec.shape[1]

    modes_3d = int_evec.reshape(N, 3, n_int)
    inv_eig = 1.0 / int_eig
    W = modes_3d * np.sqrt(inv_eig)[np.newaxis, np.newaxis, :]
    W_flat = W.reshape(N, 3 * n_int)
    cov = W_flat @ W_flat.T

    diag = np.sqrt(np.diag(cov))
    diag[diag == 0] = 1e-30
    corr = cov / np.outer(diag, diag)
    np.clip(corr, -1, 1, out=corr)
    return corr, cov


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ensure(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_fig(fig, d, name):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)


def _load_enm_data(analysis_dir, model, state):
    """Load eigenvalues, eigenvectors, resnums from analysis_dir/<model>_<state>/."""
    d = Path(analysis_dir) / f"{model}_{state}"
    return {
        "eigenvalues": np.load(d / "eigenvalues.npy"),
        "eigenvectors": np.load(d / "eigenvectors.npy"),
        "resnums": np.load(d / "resnums.npy"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# DCCM computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_correlations(analysis_dir, n_rigid_wt=None, n_rigid_mut=None):
    """Compute GNM and ANM correlation matrices for WT and MUT.

    For ANM, detects and excludes rigid-body modes automatically if
    n_rigid is not provided.

    Returns dict of correlation matrices and the shared resnums.
    """
    analysis_dir = Path(analysis_dir)
    results = {}

    # GNM: all modes are internal (ProDy strips trivial mode)
    for state in ["wt", "mut"]:
        data = _load_enm_data(analysis_dir, "gnm", state)
        corr, cov = compute_gnm_corr(data["eigenvalues"], data["eigenvectors"])
        results[f"gnm_{state}"] = corr
        print(f"  GNM {state}: corr range [{corr.min():.4f}, {corr.max():.4f}]")

    # ANM: detect and exclude rigid-body modes
    for state, n_rigid_arg in [("wt", n_rigid_wt), ("mut", n_rigid_mut)]:
        data = _load_enm_data(analysis_dir, "anm", state)
        if n_rigid_arg is None:
            n_rigid = detect_rigid_body_modes(data["eigenvalues"])
        else:
            n_rigid = n_rigid_arg
        corr, cov = compute_anm_corr(data["eigenvalues"], data["eigenvectors"], n_rigid)
        results[f"anm_{state}"] = corr
        n_int = data["eigenvalues"].shape[0] - n_rigid
        print(f"  ANM {state}: {n_rigid} rigid-body modes excluded, "
              f"{n_int} internal modes, corr range [{corr.min():.4f}, {corr.max():.4f}]")

    # Difference maps
    results["gnm_diff"] = results["gnm_mut"] - results["gnm_wt"]
    results["anm_diff"] = results["anm_mut"] - results["anm_wt"]

    for key in ["gnm_diff", "anm_diff"]:
        d = results[key]
        print(f"  {key}: range [{d.min():.6f}, {d.max():.6f}], |mean| = {np.abs(d).mean():.6f}")

    resnums = _load_enm_data(analysis_dir, "gnm", "wt")["resnums"]
    return results, resnums


# ═════════════════════════════════════════════════════════════════════════════
# Plotting — DCCM heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def plot_dccm_tripanel(corr_wt, corr_mut, resnums, mutation_pos,
                       mutation_label, model_type, out_dir):
    """Three-panel heatmap: WT, MUT, ΔCC."""
    diff = corr_mut - corr_wt
    extent = [resnums[0], resnums[-1], resnums[0], resnums[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(f"{model_type} — Dynamic Cross-Correlation ({mutation_label})",
                 fontweight="bold", fontsize=11)

    titles = ["WT", f"MUT ({mutation_label})", "ΔC (MUT − WT)"]
    data_list = [corr_wt, corr_mut, diff]

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

    _save_fig(fig, out_dir, f"dccm_{model_type.lower()}_tripanel")
    print(f"  saved dccm_{model_type.lower()}_tripanel")


def plot_dccm_diff_detail(diff, resnums, mutation_pos, mutation_label,
                          model_type, out_dir):
    """Standalone difference map with enhanced annotations."""
    extent = [resnums[0], resnums[-1], resnums[0], resnums[-1]]
    abs_max = max(np.abs(diff).max(), 0.01)
    abs_lim = np.ceil(abs_max * 20) / 20
    abs_lim = max(abs_lim, 0.05)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(diff, origin="lower", cmap="seismic",
                   vmin=-abs_lim, vmax=abs_lim,
                   aspect="equal", extent=extent)
    ax.axhline(mutation_pos, color=C_SITE, lw=0.8, ls="--", alpha=0.7)
    ax.axvline(mutation_pos, color=C_SITE, lw=0.8, ls="--", alpha=0.7)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("ΔC (MUT − WT)", fontsize=9)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_title(f"{model_type} — ΔC ({mutation_label})", fontweight="bold", pad=8)

    _save_fig(fig, out_dir, f"dccm_diff_{model_type.lower()}")
    print(f"  saved dccm_diff_{model_type.lower()}")


def plot_coupling_to_mutation(corr_wt, corr_mut, resnums, mutation_pos,
                              mutation_label, model_type, out_dir):
    """Coupling profile: correlation of each residue with mutation site."""
    idx = int(np.argmin(np.abs(resnums - mutation_pos)))
    coup_wt = corr_wt[idx, :]
    coup_mut = corr_mut[idx, :]
    delta = coup_mut - coup_wt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(resnums, coup_wt, color=C_WT, lw=0.9, alpha=0.85, label="WT")
    ax.plot(resnums, coup_mut, color=C_MUT, lw=0.9, alpha=0.85, label=mutation_label)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(mutation_pos, color=C_SITE, lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(f"C(res, site {mutation_pos})")
    ax.set_title(f"{model_type} — Coupling to Mutation Site ({mutation_label})",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    ax2.fill_between(resnums, delta, 0, where=(delta >= 0), color=C_POS, alpha=0.4)
    ax2.fill_between(resnums, delta, 0, where=(delta < 0), color=C_NEG, alpha=0.4)
    ax2.plot(resnums, delta, color=C_DELTA, lw=0.7)
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    ax2.axvline(mutation_pos, color=C_SITE, lw=0.8, ls="--", alpha=0.7)
    ax2.set_xlabel("Residue number")
    ax2.set_ylabel("ΔC (MUT − WT)")

    _save_fig(fig, out_dir, f"dccm_coupling_{model_type.lower()}")
    print(f"  saved dccm_coupling_{model_type.lower()}")


# ═════════════════════════════════════════════════════════════════════════════
# Quantitative summaries
# ═════════════════════════════════════════════════════════════════════════════

def compute_dccm_summaries(results, resnums, mutation_pos):
    """Compute quantitative DCCM summaries."""
    site_idx = int(np.argmin(np.abs(resnums - mutation_pos)))
    summaries = {}

    for model in ["gnm", "anm"]:
        c_wt = results[f"{model}_wt"]
        c_mut = results[f"{model}_mut"]
        diff = results[f"{model}_diff"]

        # Global stats
        abs_diff = np.abs(diff)
        triu = np.triu_indices_from(diff, k=1)
        triu_vals = abs_diff[triu]

        # Coupling to mutation site
        coup_wt = c_wt[site_idx, :]
        coup_mut = c_mut[site_idx, :]
        delta_coup = coup_mut - coup_wt

        # Local neighbourhood (±10 residues)
        lo = max(0, site_idx - 10)
        hi = min(len(resnums), site_idx + 11)
        local_mask = np.zeros(len(resnums), dtype=bool)
        local_mask[lo:hi] = True
        block_wt = c_wt[np.ix_(local_mask, local_mask)]
        block_mut = c_mut[np.ix_(local_mask, local_mask)]
        n_local = block_wt.shape[0]
        if n_local >= 2:
            triu_local = np.triu_indices(n_local, k=1)
            local_mean_wt = float(np.mean(block_wt[triu_local]))
            local_mean_mut = float(np.mean(block_mut[triu_local]))
        else:
            local_mean_wt = local_mean_mut = 0.0

        # Top pairs with largest |ΔCC|
        top_pair_idx = np.argsort(triu_vals)[-10:][::-1]
        top_pairs = []
        for idx in top_pair_idx:
            ri = int(resnums[triu[0][idx]])
            rj = int(resnums[triu[1][idx]])
            top_pairs.append({
                "res_i": ri, "res_j": rj,
                "delta_cc": float(diff[triu[0][idx], triu[1][idx]]),
                "abs_delta_cc": float(triu_vals[idx]),
            })

        summaries[model.upper()] = {
            "delta_cc_frobenius_norm": float(np.linalg.norm(diff, "fro")),
            "delta_cc_abs_mean": float(abs_diff.mean()),
            "delta_cc_abs_max": float(abs_diff.max()),
            "coupling_to_mutation": {
                "global_mean_wt": float(coup_wt.mean()),
                "global_mean_mut": float(coup_mut.mean()),
                "global_delta": float(delta_coup.mean()),
            },
            "local_neighbourhood": {
                "window": f"±10 residues around {mutation_pos}",
                "n_residues": n_local,
                "mean_internal_coupling_wt": round(local_mean_wt, 6),
                "mean_internal_coupling_mut": round(local_mean_mut, 6),
                "delta": round(local_mean_mut - local_mean_wt, 6),
            },
            "top10_pairs": top_pairs,
        }

    return summaries


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_dccm_analysis(
    analysis_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    mutation_label: str,
    mutation_pos: int,
) -> dict:
    """Full DCCM analysis with proper rigid-body mode exclusion."""

    print("=" * 70)
    print(f"DCCM Analysis: {mutation_label}")
    print("=" * 70)

    analysis_dir = Path(analysis_dir)
    out_dir = _ensure(out_dir)
    fig_dir = _ensure(fig_dir)

    # Compute correlations (auto-detects rigid-body modes)
    print("\n1. Computing correlation matrices (internal modes only) ...")
    corr_results, resnums = compute_correlations(analysis_dir)

    # Save correlation matrices
    for key, mat in corr_results.items():
        np.save(out_dir / f"{key}_corr.npy", mat)
    np.save(out_dir / "resnums.npy", resnums)
    print(f"  Saved {len(corr_results)} correlation matrices")

    # Plot DCCM heatmaps
    print("\n2. Generating DCCM figures ...")
    for model in ["GNM", "ANM"]:
        plot_dccm_tripanel(
            corr_results[f"{model.lower()}_wt"],
            corr_results[f"{model.lower()}_mut"],
            resnums, mutation_pos, mutation_label, model, fig_dir,
        )
        plot_dccm_diff_detail(
            corr_results[f"{model.lower()}_diff"],
            resnums, mutation_pos, mutation_label, model, fig_dir,
        )
        plot_coupling_to_mutation(
            corr_results[f"{model.lower()}_wt"],
            corr_results[f"{model.lower()}_mut"],
            resnums, mutation_pos, mutation_label, model, fig_dir,
        )

    # Quantitative summaries
    print("\n3. Computing quantitative summaries ...")
    summaries = compute_dccm_summaries(corr_results, resnums, mutation_pos)
    with open(out_dir / "dccm_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n  GNM |ΔC| Frobenius = {summaries['GNM']['delta_cc_frobenius_norm']:.4f}")
    print(f"  ANM |ΔC| Frobenius = {summaries['ANM']['delta_cc_frobenius_norm']:.4f}")

    n_figs = len(list(fig_dir.glob("dccm_*.pdf")))
    print(f"\nDCCM analysis complete: {n_figs} figures saved")
    print(f"All outputs → {out_dir}/  and  {fig_dir}/")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="DCCM analysis with rigid-body mode exclusion")
    parser.add_argument("--analysis-dir", required=True,
                        help="Analysis directory (with gnm_wt/, anm_wt/ etc.)")
    parser.add_argument("--outdir", default="dccm", help="Output directory")
    parser.add_argument("--figdir", default=None, help="Figure output directory")
    parser.add_argument("--label", required=True, help="Mutation label (e.g. V13M)")
    parser.add_argument("--site", type=int, required=True, help="Mutation position")
    args = parser.parse_args()

    fig_dir = Path(args.figdir) if args.figdir else Path(args.outdir) / "figures"
    run_dccm_analysis(
        analysis_dir=Path(args.analysis_dir),
        out_dir=Path(args.outdir),
        fig_dir=fig_dir,
        mutation_label=args.label,
        mutation_pos=args.site,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
