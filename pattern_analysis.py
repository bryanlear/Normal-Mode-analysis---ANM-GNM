#!/usr/bin/env python3
"""
Five-part advanced pattern analysis — generic pipeline module.

  1. MSF difference (per-residue mean-square fluctuation Δ)
  2. Cross-correlation comparison (ΔCC heatmaps, Frobenius norm)
  3. Eigenvector overlap (pairwise overlap matrix, RMSIP)
  4. Hinge-shift analysis (GNM zero-crossings)
  5. PRS / allosteric communication (perturbation response scanning)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import prody
from prody import ANM, GNM


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_ca(pdb: Path):
    struct = prody.parsePDB(str(pdb))
    ca = struct.select("calpha")
    assert ca is not None, f"No Cα in {pdb}"
    return ca


def build_gnm(ca, label, cutoff=11.0, n_modes=20):
    g = GNM(label)
    g.buildKirchhoff(ca, cutoff=cutoff)
    g.calcModes(n_modes=n_modes)
    return g


def build_anm(ca, label, cutoff=11.0, n_modes=20):
    a = ANM(label)
    a.buildHessian(ca, cutoff=cutoff)
    a.calcModes(n_modes=n_modes)
    return a


# ═════════════════════════════════════════════════════════════════════════════
# 1. MSF Difference
# ═════════════════════════════════════════════════════════════════════════════

def analysis_1_msf(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, mutation_pos, out):
    d = ensure(out / "1_msf_difference")
    results = {}

    for tag, model_wt, model_mut in [("gnm", gnm_wt, gnm_mut),
                                      ("anm", anm_wt, anm_mut)]:
        msf_wt = prody.calcSqFlucts(model_wt)
        msf_mut = prody.calcSqFlucts(model_mut)
        delta = msf_mut - msf_wt
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(msf_wt > 0, delta / msf_wt, 0.0)

        np.save(d / f"{tag}_msf_wt.npy", msf_wt)
        np.save(d / f"{tag}_msf_mut.npy", msf_mut)
        np.save(d / f"{tag}_delta_msf.npy", delta)
        np.save(d / f"{tag}_frac_delta_msf.npy", frac)

        site_idx = mutation_pos - int(resnums[0])
        results[tag] = {
            "msf_wt_at_site": float(msf_wt[site_idx]),
            "msf_mut_at_site": float(msf_mut[site_idx]),
            "delta_at_site": float(delta[site_idx]),
            "frac_at_site": float(frac[site_idx]),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "delta_max": float(delta.max()),
            "delta_min": float(delta.min()),
            "top5_increased": [int(resnums[i]) for i in np.argsort(delta)[-5:][::-1]],
            "top5_decreased": [int(resnums[i]) for i in np.argsort(delta)[:5]],
        }

    np.save(d / "resnums.npy", resnums)
    with open(d / "msf_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  [1] MSF difference:  saved")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. Cross-Correlation Comparison
# ═════════════════════════════════════════════════════════════════════════════

def analysis_2_crosscorr(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, mutation_pos, out):
    d = ensure(out / "2_crosscorr_comparison")
    results = {}

    for tag, model_wt, model_mut in [("gnm", gnm_wt, gnm_mut),
                                      ("anm", anm_wt, anm_mut)]:
        cc_wt = prody.calcCrossCorr(model_wt)
        cc_mut = prody.calcCrossCorr(model_mut)
        delta = cc_mut - cc_wt

        np.save(d / f"{tag}_cc_wt.npy", cc_wt)
        np.save(d / f"{tag}_cc_mut.npy", cc_mut)
        np.save(d / f"{tag}_delta_cc.npy", delta)

        mean_abs_delta = np.mean(np.abs(delta), axis=1)
        np.save(d / f"{tag}_mean_abs_delta_cc.npy", mean_abs_delta)

        site_idx = mutation_pos - int(resnums[0])
        np.save(d / f"{tag}_delta_cc_at_site.npy", delta[site_idx, :])

        # ── Top-N residue pairs with largest |ΔCC| ────────────────────
        abs_delta = np.abs(delta)
        # Upper triangle only (avoid duplicates)
        triu_mask = np.triu_indices_from(delta, k=1)
        pair_vals = abs_delta[triu_mask]
        top_pair_idx = np.argsort(pair_vals)[-10:][::-1]
        top_pairs = []
        for idx in top_pair_idx:
            ri = int(resnums[triu_mask[0][idx]])
            rj = int(resnums[triu_mask[1][idx]])
            top_pairs.append({"res_i": ri, "res_j": rj,
                              "delta_cc": float(delta[triu_mask[0][idx], triu_mask[1][idx]]),
                              "abs_delta_cc": float(pair_vals[idx])})

        # ── Local vs distal ΔCC stats ──────────────────────────────────
        LOCAL_WINDOW = 20  # residues around mutation site
        n_res = len(resnums)
        lo = max(0, site_idx - LOCAL_WINDOW)
        hi = min(n_res, site_idx + LOCAL_WINDOW + 1)
        local_delta = delta[lo:hi, lo:hi]
        distal_mask = np.ones_like(delta, dtype=bool)
        distal_mask[lo:hi, lo:hi] = False
        distal_delta = delta[distal_mask]

        # ── Fraction of pairs exceeding thresholds ─────────────────────
        n_pairs = len(pair_vals)
        frac_gt_001 = float((pair_vals > 0.01).sum() / n_pairs) if n_pairs > 0 else 0.0
        frac_gt_002 = float((pair_vals > 0.02).sum() / n_pairs) if n_pairs > 0 else 0.0
        frac_gt_005 = float((pair_vals > 0.05).sum() / n_pairs) if n_pairs > 0 else 0.0

        results[tag] = {
            "delta_cc_frobenius_norm": float(np.linalg.norm(delta, "fro")),
            "delta_cc_abs_mean": float(np.abs(delta).mean()),
            "delta_cc_abs_max": float(np.abs(delta).max()),
            "mean_abs_delta_at_site": float(mean_abs_delta[site_idx]),
            "top5_coupling_changed": [int(resnums[i]) for i in np.argsort(mean_abs_delta)[-5:][::-1]],
            "top10_pairs": top_pairs,
            "local_delta_cc_mean": float(np.mean(np.abs(local_delta))),
            "local_delta_cc_max": float(np.max(np.abs(local_delta))),
            "distal_delta_cc_mean": float(np.mean(np.abs(distal_delta))),
            "distal_delta_cc_max": float(np.max(np.abs(distal_delta))),
            "frac_pairs_gt_0.01": frac_gt_001,
            "frac_pairs_gt_0.02": frac_gt_002,
            "frac_pairs_gt_0.05": frac_gt_005,
        }

        # ── Per-mode covariance decomposition at mutation site ──────────
        # For each mode k, compute the covariance contribution between the
        # mutation site and all other residues: Cov_k(site, j) = v_k[site]·v_k[j] / λ_k
        # Then report mean|Cov_k|, percentage of total, and WT↔MUT difference.
        is_anm = (tag == "anm")
        n_pm = min(10, model_wt.numModes(), model_mut.numModes())
        per_mode_data = []
        for k in range(n_pm):
            if is_anm:
                vw = model_wt[k].getEigvec().reshape(-1, 3)
                vm = model_mut[k].getEigvec().reshape(-1, 3)
                lw = float(model_wt[k].getEigval())
                lm = float(model_mut[k].getEigval())
                cov_wt_k = np.einsum('a,ja->j', vw[site_idx], vw) / lw
                cov_mt_k = np.einsum('a,ja->j', vm[site_idx], vm) / lm
            else:
                vw = model_wt[k].getEigvec()
                vm = model_mut[k].getEigvec()
                lw = float(model_wt[k].getEigval())
                lm = float(model_mut[k].getEigval())
                cov_wt_k = vw[site_idx] * vw / lw
                cov_mt_k = vm[site_idx] * vm / lm
            per_mode_data.append({
                "mean_abs_cov_wt": float(np.mean(np.abs(cov_wt_k))),
                "mean_abs_cov_mut": float(np.mean(np.abs(cov_mt_k))),
                "cov_self_wt": float(cov_wt_k[site_idx]),
                "cov_self_mut": float(cov_mt_k[site_idx]),
            })

        tot_wt = sum(v["mean_abs_cov_wt"] for v in per_mode_data)
        tot_mt = sum(v["mean_abs_cov_mut"] for v in per_mode_data)
        per_mode_cc = {}
        for k, pm in enumerate(per_mode_data):
            pm["pct_cov_wt"] = 100 * pm["mean_abs_cov_wt"] / tot_wt if tot_wt > 0 else 0.0
            pm["pct_cov_mut"] = 100 * pm["mean_abs_cov_mut"] / tot_mt if tot_mt > 0 else 0.0
            pm["delta_mean_abs_cov"] = pm["mean_abs_cov_mut"] - pm["mean_abs_cov_wt"]
            per_mode_cc[f"mode_{k+1}"] = pm

        np.save(d / f"{tag}_per_mode_cov_at_site.npy",
                np.array([[pm["mean_abs_cov_wt"], pm["mean_abs_cov_mut"]]
                          for pm in per_mode_data]))
        results[tag]["per_mode_cc"] = per_mode_cc

    np.save(d / "resnums.npy", resnums)
    with open(d / "crosscorr_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  [2] Cross-correlation comparison:  saved")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. Eigenvector Overlap
# ═════════════════════════════════════════════════════════════════════════════

def analysis_3_overlap(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, out):
    d = ensure(out / "3_eigenvector_overlap")
    results = {}

    for tag, model_wt, model_mut in [("gnm", gnm_wt, gnm_mut),
                                      ("anm", anm_wt, anm_mut)]:
        n = min(model_wt.numModes(), model_mut.numModes())
        overlap_matrix = np.zeros((n, n))
        for i in range(n):
            vi = model_wt[i].getEigvec()
            for j in range(n):
                vj = model_mut[j].getEigvec()
                overlap_matrix[i, j] = abs(np.dot(vi, vj))

        np.save(d / f"{tag}_overlap_matrix.npy", overlap_matrix)

        diag = np.diag(overlap_matrix)
        np.save(d / f"{tag}_mode_overlaps.npy", diag)

        cum_overlap = np.sqrt(np.cumsum(overlap_matrix ** 2, axis=1))
        np.save(d / f"{tag}_cumulative_overlap.npy", cum_overlap)

        m = min(10, n)
        rmsip = np.sqrt(np.sum(overlap_matrix[:m, :m] ** 2) / m)
        np.save(d / f"{tag}_rmsip.npy", np.array([rmsip]))

        results[tag] = {
            "n_modes": int(n),
            "diagonal_overlaps": diag.tolist(),
            "mean_diagonal_overlap": float(diag.mean()),
            "min_diagonal_overlap": float(diag.min()),
            "min_overlap_mode": int(np.argmin(diag) + 1),
            "rmsip_10": float(rmsip),
        }

    np.save(d / "resnums.npy", resnums)
    with open(d / "overlap_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  [3] Eigenvector overlap:  saved")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4. Hinge-Shift Analysis
# ═════════════════════════════════════════════════════════════════════════════

def _find_zero_crossings(eigvec):
    signs = np.sign(eigvec)
    return np.where(np.diff(signs) != 0)[0]


def analysis_4_hinges(gnm_wt, gnm_mut, resnums, out, n_hinge_modes=5):
    d = ensure(out / "4_hinge_shift")
    comparisons = {}

    for k in range(min(n_hinge_modes, gnm_wt.numModes(), gnm_mut.numModes())):
        mode_label = f"mode_{k+1}"

        ev_wt = gnm_wt[k].getEigvec()
        np.save(d / f"wt_{mode_label}_eigvec.npy", ev_wt)
        zc_wt = _find_zero_crossings(ev_wt)
        hinge_resnums_wt = [int(resnums[i]) for i in zc_wt]

        ev_mut = gnm_mut[k].getEigvec()
        np.save(d / f"mut_{mode_label}_eigvec.npy", ev_mut)
        zc_mut = _find_zero_crossings(ev_mut)
        hinge_resnums_mut = [int(resnums[i]) for i in zc_mut]

        hinges_wt_prody = prody.calcHinges(gnm_wt[k])
        hinges_mut_prody = prody.calcHinges(gnm_mut[k])

        set_wt = set(hinge_resnums_wt)
        set_mut = set(hinge_resnums_mut)

        comparisons[mode_label] = {
            "wt_hinges": hinge_resnums_wt,
            "mut_hinges": hinge_resnums_mut,
            "wt_hinges_prody": [int(h) for h in hinges_wt_prody] if hinges_wt_prody is not None else [],
            "mut_hinges_prody": [int(h) for h in hinges_mut_prody] if hinges_mut_prody is not None else [],
            "conserved": sorted(set_wt & set_mut),
            "lost_in_mutant": sorted(set_wt - set_mut),
            "gained_in_mutant": sorted(set_mut - set_wt),
            "n_conserved": len(set_wt & set_mut),
            "n_lost": len(set_wt - set_mut),
            "n_gained": len(set_mut - set_wt),
        }

    np.save(d / "resnums.npy", resnums)
    summary = {"n_hinge_modes_analysed": n_hinge_modes, "per_mode": comparisons}
    with open(d / "hinge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  [4] Hinge-shift analysis:  saved")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 5. PRS / Allosteric Communication
# ═════════════════════════════════════════════════════════════════════════════

def analysis_5_prs(anm_wt, anm_mut, resnums, mutation_pos, out):
    d = ensure(out / "5_prs_allosteric")
    results = {}

    for tag, model in [("wt", anm_wt), ("mut", anm_mut)]:
        prs_matrix, effectiveness, sensitivity = prody.calcPerturbResponse(model)
        np.save(d / f"{tag}_prs_matrix.npy", prs_matrix)
        np.save(d / f"{tag}_effectiveness.npy", effectiveness)
        np.save(d / f"{tag}_sensitivity.npy", sensitivity)

        results[tag] = {
            "effectiveness_mean": float(effectiveness.mean()),
            "sensitivity_mean": float(sensitivity.mean()),
            "top5_effectors": [int(resnums[i]) for i in np.argsort(effectiveness)[-5:][::-1]],
            "top5_sensors": [int(resnums[i]) for i in np.argsort(sensitivity)[-5:][::-1]],
        }

    # N-terminus propagation
    prs_wt = np.load(d / "wt_prs_matrix.npy")
    prs_mut = np.load(d / "mut_prs_matrix.npy")
    nterm_wt = prs_wt[0, :]
    nterm_mut = prs_mut[0, :]
    delta_nterm = nterm_mut - nterm_wt

    np.save(d / "nterm_propagation_wt.npy", nterm_wt)
    np.save(d / "nterm_propagation_mut.npy", nterm_mut)
    np.save(d / "delta_nterm_propagation.npy", delta_nterm)

    results["nterm_propagation"] = {
        "source_residue": int(resnums[0]),
        "wt_mean_response": float(nterm_wt.mean()),
        "mut_mean_response": float(nterm_mut.mean()),
        "delta_mean": float(delta_nterm.mean()),
        "delta_max": float(delta_nterm.max()),
        "delta_min": float(delta_nterm.min()),
        "top5_increased_sensitivity": [int(resnums[i]) for i in np.argsort(delta_nterm)[-5:][::-1]],
        "top5_decreased_sensitivity": [int(resnums[i]) for i in np.argsort(delta_nterm)[:5]],
    }

    # Effectiveness / sensitivity deltas
    eff_wt = np.load(d / "wt_effectiveness.npy")
    eff_mut = np.load(d / "mut_effectiveness.npy")
    sen_wt = np.load(d / "wt_sensitivity.npy")
    sen_mut = np.load(d / "mut_sensitivity.npy")

    delta_eff = eff_mut - eff_wt
    delta_sen = sen_mut - sen_wt
    np.save(d / "delta_effectiveness.npy", delta_eff)
    np.save(d / "delta_sensitivity.npy", delta_sen)

    site_idx = mutation_pos - int(resnums[0])
    results["delta_prs"] = {
        "delta_effectiveness_at_site": float(delta_eff[site_idx]),
        "delta_sensitivity_at_site": float(delta_sen[site_idx]),
        "delta_eff_mean": float(delta_eff.mean()),
        "delta_sen_mean": float(delta_sen.mean()),
        "top5_eff_increase": [int(resnums[i]) for i in np.argsort(delta_eff)[-5:][::-1]],
        "top5_sen_increase": [int(resnums[i]) for i in np.argsort(delta_sen)[-5:][::-1]],
    }

    # Full ΔPRS matrix
    delta_prs = prs_mut - prs_wt
    np.save(d / "delta_prs_matrix.npy", delta_prs)

    np.save(d / "resnums.npy", resnums)
    with open(d / "prs_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  [5] PRS / allosteric communication:  saved")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def run_pattern_analysis(
    wt_pdb: Path,
    mut_pdb: Path,
    mutation_label: str,
    mutation_pos: int,
    out_dir: Path,
    gnm_cutoff: float = 11.0,
    anm_cutoff: float = 11.0,
    n_modes: int = 20,
) -> dict:
    """Run all 5 pattern analyses. Returns master results."""

    print("=" * 70)
    print(f"Pattern Analysis: {mutation_label}")
    print("=" * 70)

    out_dir = Path(out_dir)

    print("\nLoading structures & building ENMs ...")
    ca_wt = load_ca(wt_pdb)
    ca_mut = load_ca(mut_pdb)
    resnums = ca_wt.getResnums()
    print(f"  WT: {ca_wt.numAtoms()} Cα  |  MUT: {ca_mut.numAtoms()} Cα")
    print(f"  GNM cutoff: {gnm_cutoff} Å  |  ANM cutoff: {anm_cutoff} Å")

    gnm_wt = build_gnm(ca_wt, "WT_GNM", gnm_cutoff, n_modes)
    gnm_mut = build_gnm(ca_mut, "MUT_GNM", gnm_cutoff, n_modes)
    anm_wt = build_anm(ca_wt, "WT_ANM", anm_cutoff, n_modes)
    anm_mut = build_anm(ca_mut, "MUT_ANM", anm_cutoff, n_modes)

    d = ensure(out_dir)
    np.save(d / "ca_coords_wt.npy", ca_wt.getCoords())
    np.save(d / "ca_coords_mut.npy", ca_mut.getCoords())
    np.save(d / "resnums.npy", resnums)

    print("\nRunning analyses ...")
    r1 = analysis_1_msf(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, mutation_pos, out_dir)
    r2 = analysis_2_crosscorr(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, mutation_pos, out_dir)
    r3 = analysis_3_overlap(gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, out_dir)
    r4 = analysis_4_hinges(gnm_wt, gnm_mut, resnums, out_dir)
    r5 = analysis_5_prs(anm_wt, anm_mut, resnums, mutation_pos, out_dir)

    master = {
        "mutation": mutation_label,
        "mutation_position": mutation_pos,
        "cutoffs": {"gnm": gnm_cutoff, "anm": anm_cutoff},
        "n_modes": n_modes,
        "1_msf_difference": r1,
        "2_crosscorr_comparison": r2,
        "3_eigenvector_overlap": r3,
        "4_hinge_shift": r4,
        "5_prs_allosteric": r5,
    }
    with open(out_dir / "master_pattern_results.json", "w") as f:
        json.dump(master, f, indent=2)

    # ── Print summary ──
    site_idx = mutation_pos - int(resnums[0])
    print(f"\n{'='*70}")
    print(f"PATTERN ANALYSIS RESULTS — {mutation_label}")
    print(f"{'='*70}")

    print("\n── 1. MSF Difference ──")
    for tag in ("gnm", "anm"):
        m = r1[tag]
        print(f"  {tag.upper()}: ΔMSF at site = {m['delta_at_site']:+.6f} (frac = {m['frac_at_site']:+.4f})")

    print("\n── 2. Cross-Correlation Comparison ──")
    for tag in ("gnm", "anm"):
        m = r2[tag]
        print(f"  {tag.upper()}: ΔCC Frobenius norm = {m['delta_cc_frobenius_norm']:.6f}")

    print("\n── 3. Eigenvector Overlap ──")
    for tag in ("gnm", "anm"):
        m = r3[tag]
        print(f"  {tag.upper()}: mean overlap = {m['mean_diagonal_overlap']:.6f}, RMSIP = {m['rmsip_10']:.6f}")

    print("\n── 4. Hinge-Shift ──")
    for ml, mc in r4["per_mode"].items():
        print(f"  {ml}: conserved={mc['n_conserved']} lost={mc['n_lost']} gained={mc['n_gained']}")

    print("\n── 5. PRS ──")
    nt = r5["nterm_propagation"]
    print(f"  N-term Δ mean response = {nt['delta_mean']:+.6f}")

    print(f"\nAll outputs → {out_dir}/")

    return master, gnm_wt, gnm_mut, anm_wt, anm_mut, resnums


def main():
    parser = argparse.ArgumentParser(description="5-part ENM pattern analysis")
    parser.add_argument("--wt", required=True, help="Wild-type PDB")
    parser.add_argument("--mut", required=True, help="Mutant PDB")
    parser.add_argument("--label", required=True, help="Mutation label (e.g. V13M)")
    parser.add_argument("--site", type=int, required=True, help="Mutation site (residue number)")
    parser.add_argument("--outdir", default="patterns", help="Output directory")
    parser.add_argument("--gnm-cutoff", type=float, default=11.0)
    parser.add_argument("--anm-cutoff", type=float, default=11.0)
    parser.add_argument("--modes", type=int, default=20)
    args = parser.parse_args()

    run_pattern_analysis(
        wt_pdb=Path(args.wt), mut_pdb=Path(args.mut),
        mutation_label=args.label, mutation_pos=args.site,
        out_dir=Path(args.outdir),
        gnm_cutoff=args.gnm_cutoff, anm_cutoff=args.anm_cutoff,
        n_modes=args.modes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
