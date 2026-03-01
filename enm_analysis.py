#!/usr/bin/env python3
"""
GNM + ANM elastic-network-model analysis — generic pipeline module.

Loads WT and mutant PDB structures, determines optimal cutoffs via
Minimum Connectivity Threshold (MCT), runs GNM and ANM for both,
and computes comparative metrics (ΔSqFluct, mode overlaps, ΔCC).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import prody
from prody import ANM, GNM


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_calpha(pdb_path: Path):
    """Parse PDB → (full structure, Cα AtomGroup)."""
    structure = prody.parsePDB(str(pdb_path))
    ca = structure.select("calpha")
    if ca is None:
        raise RuntimeError(f"No Cα atoms found in {pdb_path}")
    return structure, ca


# ── Minimum Connectivity Threshold ──────────────────────────────────────────

def find_mct_cutoff(ca_atoms, model_type="GNM",
                    scan_range=None, step=0.5, min_contacts=3):
    """Sweep cutoffs to find the minimum where every node has ≥ min_contacts neighbours.

    Returns dict with cutoff (Å), stats, and full scan log.
    """
    if scan_range is None:
        scan_range = (5.0, 12.0) if model_type.upper() == "GNM" else (10.0, 18.0)

    results = []
    chosen = None
    start, stop = scan_range
    c = start
    while c <= stop + 1e-9:
        if model_type.upper() == "GNM":
            m = GNM("scan")
            m.buildKirchhoff(ca_atoms, cutoff=c)
            K = m.getKirchhoff()
            diag = np.diag(K)
        else:
            m = ANM("scan")
            m.buildHessian(ca_atoms, cutoff=c)
            H = m.getHessian()
            n = ca_atoms.numAtoms()
            diag = np.array([np.trace(H[3*i:3*i+3, 3*i:3*i+3]) for i in range(n)])

        mc = int(diag.min())
        mean_c = float(diag.mean())
        disc = int((diag == 0).sum())
        results.append({
            "cutoff": round(c, 2),
            "min_contacts": mc,
            "mean_contacts": round(mean_c, 2),
            "disconnected": disc,
        })
        if chosen is None and mc >= min_contacts:
            chosen = round(c, 2)
        c += step

    # Safety margin
    if chosen is not None:
        safe = round(chosen + step, 2)
        if safe <= stop:
            chosen = safe
    if chosen is None:
        chosen = round(stop, 2)
        print(f"  WARNING: full connectivity not reached; using max cutoff {chosen} Å")

    entry = next(e for e in results if abs(e["cutoff"] - chosen) < 1e-6)
    return {
        "cutoff": chosen,
        "min_contacts_at_cutoff": entry["min_contacts"],
        "mean_contacts_at_cutoff": entry["mean_contacts"],
        "scan": results,
    }


# ── GNM ──────────────────────────────────────────────────────────────────────

def run_gnm(ca_atoms, label, cutoff, out_dir, n_modes=20):
    """Build GNM, solve, and save all raw data. Returns (gnm, summary_dict)."""
    d = ensure_dir(out_dir)

    gnm = GNM(label)
    gnm.buildKirchhoff(ca_atoms, cutoff=cutoff)
    gnm.calcModes(n_modes=n_modes)

    np.save(d / "kirchhoff.npy", gnm.getKirchhoff())
    np.save(d / "ca_coords.npy", ca_atoms.getCoords())

    eigenvalues = gnm.getEigvals()
    eigenvectors = gnm.getEigvecs()
    np.save(d / "eigenvalues.npy", eigenvalues)
    np.save(d / "eigenvectors.npy", eigenvectors)

    sq_flucts = prody.calcSqFlucts(gnm)
    np.save(d / "sqflucts.npy", sq_flucts)

    cross_corr = prody.calcCrossCorr(gnm)
    np.save(d / "cross_correlation.npy", cross_corr)

    for i in range(min(10, n_modes)):
        np.save(d / f"mode_{i+1}_shape.npy", gnm[i].getEigvec())

    resnums = ca_atoms.getResnums()
    np.save(d / "resnums.npy", resnums)

    summary = {
        "label": label,
        "model": "GNM",
        "cutoff_A": cutoff,
        "n_residues": int(ca_atoms.numAtoms()),
        "n_modes_computed": n_modes,
        "eigenvalues": eigenvalues.tolist(),
        "sq_flucts_mean": float(sq_flucts.mean()),
        "sq_flucts_std": float(sq_flucts.std()),
    }
    with open(d / "gnm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  GNM [{label}] done: {n_modes} modes, cutoff={cutoff} Å")
    return gnm, summary


# ── ANM ──────────────────────────────────────────────────────────────────────

def run_anm(ca_atoms, label, cutoff, out_dir, n_modes=20):
    """Build ANM, solve, and save all raw data. Returns (anm, summary_dict)."""
    d = ensure_dir(out_dir)

    anm = ANM(label)
    anm.buildHessian(ca_atoms, cutoff=cutoff)
    anm.calcModes(n_modes=n_modes)

    np.save(d / "hessian.npy", anm.getHessian())
    np.save(d / "ca_coords.npy", ca_atoms.getCoords())

    eigenvalues = anm.getEigvals()
    eigenvectors = anm.getEigvecs()
    np.save(d / "eigenvalues.npy", eigenvalues)
    np.save(d / "eigenvectors.npy", eigenvectors)

    sq_flucts = prody.calcSqFlucts(anm)
    np.save(d / "sqflucts.npy", sq_flucts)

    cross_corr = prody.calcCrossCorr(anm)
    np.save(d / "cross_correlation.npy", cross_corr)

    for i in range(min(10, n_modes)):
        np.save(d / f"mode_{i+1}_shape.npy", anm[i].getEigvec())

    resnums = ca_atoms.getResnums()
    np.save(d / "resnums.npy", resnums)

    summary = {
        "label": label,
        "model": "ANM",
        "cutoff_A": cutoff,
        "n_residues": int(ca_atoms.numAtoms()),
        "n_modes_computed": n_modes,
        "eigenvalues": eigenvalues.tolist(),
        "sq_flucts_mean": float(sq_flucts.mean()),
        "sq_flucts_std": float(sq_flucts.std()),
    }
    with open(d / "anm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  ANM [{label}] done: {n_modes} modes, cutoff={cutoff} Å")
    return anm, summary


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_fluctuations(gnm_wt, gnm_mut, anm_wt, anm_mut,
                         resnums, mutation_label, mutation_pos, out_dir):
    """ΔSqFluct, mode overlaps, ΔCC — save all comparison arrays."""
    d = ensure_dir(out_dir)

    # ── GNM ──
    sf_wt_g = prody.calcSqFlucts(gnm_wt)
    sf_mut_g = prody.calcSqFlucts(gnm_mut)
    delta_g = sf_mut_g - sf_wt_g
    np.save(d / "gnm_delta_sqflucts.npy", delta_g)
    np.save(d / "gnm_sqflucts_wt.npy", sf_wt_g)
    np.save(d / "gnm_sqflucts_mut.npy", sf_mut_g)

    # ── ANM ──
    sf_wt_a = prody.calcSqFlucts(anm_wt)
    sf_mut_a = prody.calcSqFlucts(anm_mut)
    delta_a = sf_mut_a - sf_wt_a
    np.save(d / "anm_delta_sqflucts.npy", delta_a)
    np.save(d / "anm_sqflucts_wt.npy", sf_wt_a)
    np.save(d / "anm_sqflucts_mut.npy", sf_mut_a)

    # ── Cross-correlation difference ──
    cc_wt_g = prody.calcCrossCorr(gnm_wt)
    cc_mut_g = prody.calcCrossCorr(gnm_mut)
    np.save(d / "gnm_delta_crosscorr.npy", cc_mut_g - cc_wt_g)

    cc_wt_a = prody.calcCrossCorr(anm_wt)
    cc_mut_a = prody.calcCrossCorr(anm_mut)
    np.save(d / "anm_delta_crosscorr.npy", cc_mut_a - cc_wt_a)

    # ── Mode overlaps (first 10 modes) ──
    gnm_overlaps, anm_overlaps = [], []
    for i in range(min(10, gnm_wt.numModes(), gnm_mut.numModes())):
        gnm_overlaps.append(float(abs(np.dot(gnm_wt[i].getEigvec(), gnm_mut[i].getEigvec()))))
    for i in range(min(10, anm_wt.numModes(), anm_mut.numModes())):
        anm_overlaps.append(float(abs(np.dot(anm_wt[i].getEigvec(), anm_mut[i].getEigvec()))))

    np.save(d / "gnm_mode_overlaps.npy", np.array(gnm_overlaps))
    np.save(d / "anm_mode_overlaps.npy", np.array(anm_overlaps))

    # ── Subspace overlap (first 10 modes) ──
    gnm_subspace = prody.calcOverlap(gnm_wt[:10], gnm_mut[:10])
    anm_subspace = prody.calcOverlap(anm_wt[:10], anm_mut[:10])
    np.save(d / "gnm_subspace_overlap.npy", gnm_subspace)
    np.save(d / "anm_subspace_overlap.npy", anm_subspace)

    np.save(d / "resnums.npy", resnums)

    site_idx = mutation_pos - int(resnums[0])

    stats = {
        "mutation": mutation_label,
        "mutation_position": mutation_pos,
        "gnm": {
            "delta_sqflucts_at_site": float(delta_g[site_idx]),
            "delta_sqflucts_mean": float(delta_g.mean()),
            "delta_sqflucts_std": float(delta_g.std()),
            "delta_sqflucts_max": float(delta_g.max()),
            "delta_sqflucts_min": float(delta_g.min()),
            "mode_overlaps_1_10": gnm_overlaps,
            "mean_mode_overlap": float(np.mean(gnm_overlaps)),
        },
        "anm": {
            "delta_sqflucts_at_site": float(delta_a[site_idx]),
            "delta_sqflucts_mean": float(delta_a.mean()),
            "delta_sqflucts_std": float(delta_a.std()),
            "delta_sqflucts_max": float(delta_a.max()),
            "delta_sqflucts_min": float(delta_a.min()),
            "mode_overlaps_1_10": anm_overlaps,
            "mean_mode_overlap": float(np.mean(anm_overlaps)),
        },
    }
    with open(d / "comparison_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_enm_analysis(
    wt_pdb: Path,
    mut_pdb: Path,
    mutation_label: str,
    mutation_pos: int,
    out_dir: Path,
    n_modes: int = 20,
) -> dict:
    """Full GNM+ANM analysis pipeline. Returns master results dict."""

    print("=" * 70)
    print(f"ENM Analysis: {mutation_label}")
    print("=" * 70)

    out_dir = Path(out_dir)

    # ── Load ──
    print("\n1. Loading structures ...")
    _, ca_wt = load_calpha(wt_pdb)
    _, ca_mut = load_calpha(mut_pdb)
    print(f"   WT:  {ca_wt.numAtoms()} Cα atoms")
    print(f"   MUT: {ca_mut.numAtoms()} Cα atoms")
    resnums = ca_wt.getResnums()

    # ── MCT cutoffs ──
    print("\n2. Determining optimal cutoffs (MCT) ...")
    gnm_mct = find_mct_cutoff(ca_wt, model_type="GNM")
    gnm_cutoff = gnm_mct["cutoff"]
    print(f"   GNM cutoff: {gnm_cutoff} Å  (min contacts={gnm_mct['min_contacts_at_cutoff']})")

    anm_mct = find_mct_cutoff(ca_wt, model_type="ANM")
    anm_cutoff = anm_mct["cutoff"]
    print(f"   ANM cutoff: {anm_cutoff} Å  (min contacts={anm_mct['min_contacts_at_cutoff']})")

    mct_dir = ensure_dir(out_dir / "mct_scan")
    with open(mct_dir / "mct_results.json", "w") as f:
        json.dump({"gnm": gnm_mct, "anm": anm_mct}, f, indent=2)

    # ── GNM ──
    print("\n3. Running GNM analysis ...")
    gnm_wt, gnm_wt_sum = run_gnm(ca_wt, "WT_GNM", gnm_cutoff, out_dir / "gnm_wt", n_modes)
    gnm_mut, gnm_mut_sum = run_gnm(ca_mut, "MUT_GNM", gnm_cutoff, out_dir / "gnm_mut", n_modes)

    # ── ANM ──
    print("\n4. Running ANM analysis ...")
    anm_wt, anm_wt_sum = run_anm(ca_wt, "WT_ANM", anm_cutoff, out_dir / "anm_wt", n_modes)
    anm_mut, anm_mut_sum = run_anm(ca_mut, "MUT_ANM", anm_cutoff, out_dir / "anm_mut", n_modes)

    # ── Comparison ──
    print("\n5. Computing WT vs Mutant comparison ...")
    comp = compare_fluctuations(
        gnm_wt, gnm_mut, anm_wt, anm_mut,
        resnums, mutation_label, mutation_pos,
        out_dir / "comparison",
    )

    # ── Print summary ──
    site_idx = mutation_pos - int(resnums[0])
    sf_wt_g = prody.calcSqFlucts(gnm_wt)
    sf_mut_g = prody.calcSqFlucts(gnm_mut)
    sf_wt_a = prody.calcSqFlucts(anm_wt)
    sf_mut_a = prody.calcSqFlucts(anm_mut)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — {mutation_label}")
    print(f"{'='*70}")
    print(f"GNM cutoff: {gnm_cutoff} Å  |  ANM cutoff: {anm_cutoff} Å")
    print(f"\nGNM sq. fluct. at site {mutation_pos}:")
    print(f"  WT={sf_wt_g[site_idx]:.6f}  MUT={sf_mut_g[site_idx]:.6f}  Δ={sf_mut_g[site_idx]-sf_wt_g[site_idx]:+.6f}")
    print(f"ANM sq. fluct. at site {mutation_pos}:")
    print(f"  WT={sf_wt_a[site_idx]:.6f}  MUT={sf_mut_a[site_idx]:.6f}  Δ={sf_mut_a[site_idx]-sf_wt_a[site_idx]:+.6f}")
    print(f"GNM mode overlaps: {[f'{o:.4f}' for o in comp['gnm']['mode_overlaps_1_10']]}")
    print(f"ANM mode overlaps: {[f'{o:.4f}' for o in comp['anm']['mode_overlaps_1_10']]}")
    print(f"GNM mean overlap: {comp['gnm']['mean_mode_overlap']:.4f}")
    print(f"ANM mean overlap: {comp['anm']['mean_mode_overlap']:.4f}")

    # ── Master JSON ──
    master = {
        "mutation": mutation_label,
        "mutation_position": mutation_pos,
        "pdb_wt": str(wt_pdb),
        "pdb_mut": str(mut_pdb),
        "cutoffs": {
            "gnm": {"selected": gnm_cutoff, "mct_scan": gnm_mct["scan"]},
            "anm": {"selected": anm_cutoff, "mct_scan": anm_mct["scan"]},
        },
        "gnm_wt_summary": gnm_wt_sum,
        "gnm_mut_summary": gnm_mut_sum,
        "anm_wt_summary": anm_wt_sum,
        "anm_mut_summary": anm_mut_sum,
        "comparison": comp,
    }
    with open(out_dir / "master_results.json", "w") as f:
        json.dump(master, f, indent=2)

    print(f"\nAll outputs → {out_dir}/")
    return master, gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, gnm_cutoff, anm_cutoff


def main():
    parser = argparse.ArgumentParser(description="GNM + ANM elastic-network analysis")
    parser.add_argument("--wt", required=True, help="Wild-type PDB file")
    parser.add_argument("--mut", required=True, help="Mutant PDB file")
    parser.add_argument("--label", required=True, help="Mutation label (e.g. V13M)")
    parser.add_argument("--site", type=int, required=True, help="Mutation site (PDB residue number)")
    parser.add_argument("--outdir", default="analysis", help="Output directory")
    parser.add_argument("--modes", type=int, default=20, help="Number of modes (default: 20)")
    args = parser.parse_args()

    run_enm_analysis(
        wt_pdb=Path(args.wt),
        mut_pdb=Path(args.mut),
        mutation_label=args.label,
        mutation_pos=args.site,
        out_dir=Path(args.outdir),
        n_modes=args.modes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
