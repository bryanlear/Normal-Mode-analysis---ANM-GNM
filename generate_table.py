#!/usr/bin/env python3
"""
Step 6 — Generate a comprehensive LaTeX summary table from pipeline results.

Reads master_results.json (ENM), master_pattern_results.json (patterns),
and rosetta_results.json (Rosetta ΔΔG) to produce a multi-section LaTeX
table with all key numerical statistics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(val, fmt=".4f"):
    """Format a float, returning '—' for None/missing."""
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _sf(val, fmt="+.4f"):
    """Signed float format."""
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _lst(vals, n=5, fmt=".0f"):
    """Format a list of numbers into a comma-separated string."""
    if not vals:
        return "—"
    return ", ".join(f"{v:{fmt}}" for v in vals[:n])


def _esc(s):
    """Escape LaTeX special characters in a string."""
    if not isinstance(s, str):
        return str(s)
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(ch, f"\\{ch}")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Table builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(
    analysis_dir: Path,
    pattern_dir: Path,
    outdir: Path,
    rosetta_json: Path = None,
    mutation_label: str = "?",
    mutation_pos: int = 0,
    n_modes: int = 20,
) -> Path:
    """Build a LaTeX file with a thorough numerical summary table.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing master_results.json (from enm_analysis).
    pattern_dir : Path
        Directory containing master_pattern_results.json (from pattern_analysis).
    outdir : Path
        Output directory (the top-level pipeline output).
    rosetta_json : Path, optional
        Path to rosetta_results.json.
    mutation_label : str
        e.g. "R2103K".
    mutation_pos : int
        Residue number.
    n_modes : int
        Number of modes used in analysis.

    Returns
    -------
    Path to the generated .tex file.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load available data ──────────────────────────────────────────────
    enm = {}
    enm_file = analysis_dir / "master_results.json"
    if enm_file.exists():
        with open(enm_file) as f:
            enm = json.load(f)

    pat = {}
    pat_file = pattern_dir / "master_pattern_results.json"
    if pat_file.exists():
        with open(pat_file) as f:
            pat = json.load(f)

    ros = {}
    if rosetta_json and Path(rosetta_json).exists():
        with open(rosetta_json) as f:
            ros = json.load(f)

    # ── Convenience accessors ────────────────────────────────────────────
    comp = enm.get("comparison", {})
    comp_gnm = comp.get("gnm", {})
    comp_anm = comp.get("anm", {})

    gnm_wt = enm.get("gnm_wt_summary", {})
    gnm_mut = enm.get("gnm_mut_summary", {})
    anm_wt = enm.get("anm_wt_summary", {})
    anm_mut = enm.get("anm_mut_summary", {})

    r1 = pat.get("1_msf_difference", {})
    r1g = r1.get("gnm", {})
    r1a = r1.get("anm", {})

    r2 = pat.get("2_crosscorr_comparison", {})
    r2g = r2.get("gnm", {})
    r2a = r2.get("anm", {})

    r3 = pat.get("3_eigenvector_overlap", {})
    r3g = r3.get("gnm", {})
    r3a = r3.get("anm", {})

    r4 = pat.get("4_hinge_shift", {})

    r5 = pat.get("5_prs_allosteric", {})
    r5_wt = r5.get("wt", {})
    r5_mut = r5.get("mut", {})
    r5_dprs = r5.get("delta_prs", {})
    r5_nterm = r5.get("nterm_propagation", {})

    cutoffs = enm.get("cutoffs", {})
    gnm_cut = cutoffs.get("gnm", {}).get("selected", "—")
    anm_cut = cutoffs.get("anm", {}).get("selected", "—")

    n_res = gnm_wt.get("n_residues", anm_wt.get("n_residues", "—"))

    # ── Load raw sqfluct arrays and compute z-scores ──────────────────
    # Z-score normalises fluctuations so that values dominated by a
    # handful of soft modes (common in ANM) become interpretable.
    zs = {}   # zs["gnm_wt"] = {"z_at_site": ..., "top5_high": [...], ...}
    resnums_arr = None
    for tag, subdir_wt, subdir_mut in [
        ("gnm", "gnm_wt", "gnm_mut"),
        ("anm", "anm_wt", "anm_mut"),
    ]:
        for state, subdir in [("wt", subdir_wt), ("mut", subdir_mut)]:
            key = f"{tag}_{state}"
            sf_path = analysis_dir / subdir / "sqflucts.npy"
            rn_path = analysis_dir / subdir / "resnums.npy"
            if sf_path.exists():
                sf = np.load(sf_path)
                if rn_path.exists() and resnums_arr is None:
                    resnums_arr = np.load(rn_path)
                rn = resnums_arr if resnums_arr is not None else np.arange(1, len(sf) + 1)
                mu, sigma = sf.mean(), sf.std()
                z = (sf - mu) / sigma if sigma > 0 else np.zeros_like(sf)
                site_idx = int(np.argmin(np.abs(rn - mutation_pos)))
                top5_hi_idx = np.argsort(sf)[-5:][::-1]
                top5_lo_idx = np.argsort(sf)[:5]
                zs[key] = {
                    "z_at_site": float(z[site_idx]),
                    "sf_at_site": float(sf[site_idx]),
                    "top5_high_res": [int(rn[i]) for i in top5_hi_idx],
                    "top5_high_val": [float(sf[i]) for i in top5_hi_idx],
                    "top5_high_z":   [float(z[i]) for i in top5_hi_idx],
                    "top5_low_res":  [int(rn[i]) for i in top5_lo_idx],
                    "top5_low_val":  [float(sf[i]) for i in top5_lo_idx],
                    "top5_low_z":    [float(z[i]) for i in top5_lo_idx],
                }
            else:
                zs[key] = {}

    # Also compute z-scores for ΔSqFluct (= ΔMSF)
    for tag in ["gnm", "anm"]:
        delta_path = analysis_dir / "comparison" / f"{tag}_delta_sqflucts.npy"
        if delta_path.exists():
            delta = np.load(delta_path)
            rn = resnums_arr if resnums_arr is not None else np.arange(1, len(delta) + 1)
            mu, sigma = delta.mean(), delta.std()
            z = (delta - mu) / sigma if sigma > 0 else np.zeros_like(delta)
            site_idx = int(np.argmin(np.abs(rn - mutation_pos)))
            top5_inc_idx = np.argsort(delta)[-5:][::-1]
            top5_dec_idx = np.argsort(delta)[:5]
            zs[f"{tag}_delta"] = {
                "z_at_site": float(z[site_idx]),
                "top5_inc_res": [int(rn[i]) for i in top5_inc_idx],
                "top5_inc_z":   [float(z[i]) for i in top5_inc_idx],
                "top5_dec_res": [int(rn[i]) for i in top5_dec_idx],
                "top5_dec_z":   [float(z[i]) for i in top5_dec_idx],
            }
        else:
            zs[f"{tag}_delta"] = {}

    # ── Per-mode variance decomposition ────────────────────────────────
    # Decompose total SqFluct into individual mode contributions so the
    # user can see which modes dominate.
    mode_decomp = {}  # mode_decomp["gnm_wt"] = {"pct_global": [...], "pct_at_site": [...], ...}
    rn = resnums_arr if resnums_arr is not None else None
    site_idx_md = None
    for tag, subdir_wt, subdir_mut in [
        ("gnm", "gnm_wt", "gnm_mut"),
        ("anm", "anm_wt", "anm_mut"),
    ]:
        is_anm = (tag == "anm")
        for state, subdir in [("wt", subdir_wt), ("mut", subdir_mut)]:
            key = f"{tag}_{state}"
            ev_path = analysis_dir / subdir / "eigenvalues.npy"
            ec_path = analysis_dir / subdir / "eigenvectors.npy"
            sf_path = analysis_dir / subdir / "sqflucts.npy"
            if not (ev_path.exists() and ec_path.exists() and sf_path.exists()):
                mode_decomp[key] = {}
                continue
            ev = np.load(ev_path)
            ec = np.load(ec_path)
            sf = np.load(sf_path)
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

            mode_var = per_mode_sf.sum(axis=1)    # total variance per mode
            total_var = mode_var.sum()
            pct_global = 100.0 * mode_var / total_var if total_var > 0 else np.zeros(n_m)

            # Identify site index
            rn_local = rn if rn is not None else np.arange(1, n_r + 1)
            sidx = int(np.argmin(np.abs(rn_local - mutation_pos)))
            if site_idx_md is None:
                site_idx_md = sidx
            sf_at_site_total = sf[sidx]
            pct_site = 100.0 * per_mode_sf[:, sidx] / sf_at_site_total if sf_at_site_total > 0 else np.zeros(n_m)

            # Top-contributing mode at site
            dom_mode = int(np.argmax(per_mode_sf[:, sidx])) + 1

            mode_decomp[key] = {
                "eigenvalues": ev.tolist(),
                "pct_global": pct_global.tolist(),
                "pct_site": pct_site.tolist(),
                "sf_site_per_mode": per_mode_sf[:, sidx].tolist(),
                "cum_global": np.cumsum(pct_global).tolist(),
                "dominant_mode_at_site": dom_mode,
                "dominant_mode_pct": float(pct_site[dom_mode - 1]) if len(pct_site) >= dom_mode else 0.0,
                "n_modes": n_m,
            }

    # ── Hinge summary ────────────────────────────────────────────────────
    hinge_modes = r4.get("per_mode", {})
    total_conserved = sum(m.get("n_conserved", 0) for m in hinge_modes.values())
    total_lost = sum(m.get("n_lost", 0) for m in hinge_modes.values())
    total_gained = sum(m.get("n_gained", 0) for m in hinge_modes.values())
    n_hinge_modes = r4.get("n_hinge_modes_analysed", len(hinge_modes))

    # Per-mode hinge rows
    hinge_rows = []
    for k in sorted(hinge_modes.keys()):
        m = hinge_modes[k]
        mode_num = k.replace("mode_", "")
        hinge_rows.append(
            f"        Mode {mode_num} & {m.get('n_conserved', 0)} "
            f"& {m.get('n_lost', 0)} & {m.get('n_gained', 0)} "
            f"& {_lst(m.get('conserved', []))} \\\\"
        )

    # ── Build LaTeX ──────────────────────────────────────────────────────
    tex_lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[landscape, margin=0.75in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{caption}",
        r"\usepackage{longtable}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{amsmath}",
        r"",
        r"\definecolor{secbg}{HTML}{E8EDF2}",
        r"\newcommand{\secrow}[1]{\rowcolor{secbg}\multicolumn{3}{l}{\textbf{#1}} \\}",
        r"\newcommand{\secrowfour}[1]{\rowcolor{secbg}\multicolumn{4}{l}{\textbf{#1}} \\}",
        r"\newcommand{\secrowfive}[1]{\rowcolor{secbg}\multicolumn{5}{l}{\textbf{#1}} \\}",
        r"",
        r"\begin{document}",
        r"\pagestyle{empty}",
        r"",
        r"\begin{center}",
        rf"\textbf{{\Large Mutation Dynamics Summary: {_esc(mutation_label)}}}\\[4pt]",
        rf"Mutation site: residue {mutation_pos} \quad"
        rf"$\mid$ \quad $N_{{\mathrm{{res}}}}$ = {n_res} \quad"
        rf"$\mid$ \quad Modes: {n_modes}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 1 — Master summary (3-column: Metric | GNM | ANM)
    # ══════════════════════════════════════════════════════════════════════
    tex_lines += [
        r"% ────────────────────────────────────────────────────────────",
        r"% Table 1: Master Summary",
        r"% ────────────────────────────────────────────────────────────",
        r"{\small",
        r"\begin{longtable}{p{5.2cm} r r}",
        r"\caption{Comprehensive numerical summary for mutation "
        + _esc(mutation_label) + r".} \\",
        r"\toprule",
        r"\textbf{Metric} & \textbf{GNM} & \textbf{ANM} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Metric} & \textbf{GNM} & \textbf{ANM} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{3}{r}{\textit{continued \ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        r"",
    ]

    # ── Section: Rosetta Energetics ──
    if ros:
        tex_lines += [
            r"\secrow{Rosetta Energetics}",
            r"\midrule",
            rf"Protocol & \multicolumn{{2}}{{c}}{{{_esc(ros.get('protocol', '—'))}}} \\",
            rf"WT total energy (REU) & \multicolumn{{2}}{{c}}{{{_f(ros.get('wt_energy'), '.2f')}}} \\",
            rf"MUT total energy (REU) & \multicolumn{{2}}{{c}}{{{_f(ros.get('mut_energy'), '.2f')}}} \\",
            rf"$\Delta\Delta G$ (REU) & \multicolumn{{2}}{{c}}{{{_sf(ros.get('ddg'), '+.2f')}}} \\",
            rf"Interpretation & \multicolumn{{2}}{{c}}{{{_esc(ros.get('interpretation', '—'))}}} \\",
            r"\midrule",
            r"",
        ]

    # ── Section: ENM Cutoffs ──
    tex_lines += [
        r"\secrow{Elastic Network Model Parameters}",
        r"\midrule",
        rf"Cutoff (\AA) & {_f(gnm_cut, '.1f')} & {_f(anm_cut, '.1f')} \\",
        rf"Residues & {gnm_wt.get('n_residues', '—')} & {anm_wt.get('n_residues', '—')} \\",
        rf"Modes computed & {gnm_wt.get('n_modes_computed', '—')} & {anm_wt.get('n_modes_computed', '—')} \\",
        r"\midrule",
        r"",
    ]

    # ── Section: Squared Fluctuations ──
    _gw = zs.get("gnm_wt", {})
    _gm = zs.get("gnm_mut", {})
    _aw = zs.get("anm_wt", {})
    _am = zs.get("anm_mut", {})
    _gd = zs.get("gnm_delta", {})
    _ad = zs.get("anm_delta", {})

    tex_lines += [
        r"\secrow{Squared Fluctuations}",
        r"\midrule",
        rf"WT mean SqFluct & {_f(gnm_wt.get('sq_flucts_mean'))} & {_f(anm_wt.get('sq_flucts_mean'))} \\",
        rf"WT std SqFluct & {_f(gnm_wt.get('sq_flucts_std'))} & {_f(anm_wt.get('sq_flucts_std'))} \\",
        rf"MUT mean SqFluct & {_f(gnm_mut.get('sq_flucts_mean'))} & {_f(anm_mut.get('sq_flucts_mean'))} \\",
        rf"MUT std SqFluct & {_f(gnm_mut.get('sq_flucts_std'))} & {_f(anm_mut.get('sq_flucts_std'))} \\",
        rf"WT SqFluct at site {mutation_pos} & {_f(_gw.get('sf_at_site'))} & {_f(_aw.get('sf_at_site'))} \\",
        rf"WT $z$-score at site {mutation_pos} & {_sf(_gw.get('z_at_site'))} & {_sf(_aw.get('z_at_site'))} \\",
        rf"MUT SqFluct at site {mutation_pos} & {_f(_gm.get('sf_at_site'))} & {_f(_am.get('sf_at_site'))} \\",
        rf"MUT $z$-score at site {mutation_pos} & {_sf(_gm.get('z_at_site'))} & {_sf(_am.get('z_at_site'))} \\",
        rf"$\Delta$SqFluct at site {mutation_pos} & {_sf(comp_gnm.get('delta_sqflucts_at_site'))} & {_sf(comp_anm.get('delta_sqflucts_at_site'))} \\",
        rf"$\Delta$SqFluct $z$-score at site & {_sf(_gd.get('z_at_site'))} & {_sf(_ad.get('z_at_site'))} \\",
        rf"$\Delta$SqFluct mean & {_sf(comp_gnm.get('delta_sqflucts_mean'))} & {_sf(comp_anm.get('delta_sqflucts_mean'))} \\",
        rf"$\Delta$SqFluct std & {_f(comp_gnm.get('delta_sqflucts_std'))} & {_f(comp_anm.get('delta_sqflucts_std'))} \\",
        rf"$\Delta$SqFluct max & {_sf(comp_gnm.get('delta_sqflucts_max'))} & {_sf(comp_anm.get('delta_sqflucts_max'))} \\",
        rf"$\Delta$SqFluct min & {_sf(comp_gnm.get('delta_sqflucts_min'))} & {_sf(comp_anm.get('delta_sqflucts_min'))} \\",
        r"\midrule",
    ]

    # Top/bottom residues — one row per model to avoid overflow
    for state_label, gkey, akey in [("WT", "gnm_wt", "anm_wt"), ("MUT", "gnm_mut", "anm_mut")]:
        gs = zs.get(gkey, {})
        asx = zs.get(akey, {})

        def _res_z_list(vals, zvals):
            if not vals:
                return "\u2014"
            return ", ".join(f"{r}\\,({z:+.1f})" for r, z in zip(vals, zvals))

        # GNM highest / lowest
        tex_lines.append(
            rf"{state_label} GNM top 5 high SqFluct & \multicolumn{{2}}{{r}}{{{_res_z_list(gs.get('top5_high_res'), gs.get('top5_high_z', []))}}} \\"
        )
        tex_lines.append(
            rf"{state_label} GNM top 5 low SqFluct & \multicolumn{{2}}{{r}}{{{_res_z_list(gs.get('top5_low_res'), gs.get('top5_low_z', []))}}} \\"
        )
        # ANM highest / lowest
        tex_lines.append(
            rf"{state_label} ANM top 5 high SqFluct & \multicolumn{{2}}{{r}}{{{_res_z_list(asx.get('top5_high_res'), asx.get('top5_high_z', []))}}} \\"
        )
        tex_lines.append(
            rf"{state_label} ANM top 5 low SqFluct & \multicolumn{{2}}{{r}}{{{_res_z_list(asx.get('top5_low_res'), asx.get('top5_low_z', []))}}} \\"
        )

    tex_lines += [
        r"\midrule",
        r"",
    ]

    # ── Section: MSF Difference (pattern 1) ──
    tex_lines += [
        r"\secrow{1. MSF Difference}",
        r"\midrule",
        rf"WT MSF at site {mutation_pos} & {_f(r1g.get('msf_wt_at_site'))} & {_f(r1a.get('msf_wt_at_site'))} \\",
        rf"WT $z$-score at site {mutation_pos} & {_sf(_gw.get('z_at_site'))} & {_sf(_aw.get('z_at_site'))} \\",
        rf"MUT MSF at site {mutation_pos} & {_f(r1g.get('msf_mut_at_site'))} & {_f(r1a.get('msf_mut_at_site'))} \\",
        rf"MUT $z$-score at site {mutation_pos} & {_sf(_gm.get('z_at_site'))} & {_sf(_am.get('z_at_site'))} \\",
        rf"$\Delta$MSF at site & {_sf(r1g.get('delta_at_site'))} & {_sf(r1a.get('delta_at_site'))} \\",
        rf"$\Delta$MSF $z$-score at site & {_sf(_gd.get('z_at_site'))} & {_sf(_ad.get('z_at_site'))} \\",
        rf"Fractional $\Delta$MSF at site & {_sf(r1g.get('frac_at_site'))} & {_sf(r1a.get('frac_at_site'))} \\",
        rf"$\Delta$MSF mean (global) & {_sf(r1g.get('delta_mean'))} & {_sf(r1a.get('delta_mean'))} \\",
        rf"$\Delta$MSF std & {_f(r1g.get('delta_std'))} & {_f(r1a.get('delta_std'))} \\",
        rf"$\Delta$MSF max & {_sf(r1g.get('delta_max'))} & {_sf(r1a.get('delta_max'))} \\",
        rf"$\Delta$MSF min & {_sf(r1g.get('delta_min'))} & {_sf(r1a.get('delta_min'))} \\",
        r"\midrule",
    ]

    # Top 5 increased / decreased with ΔMSF z-scores
    def _res_z_list_delta(vals, zvals):
        if not vals:
            return "\u2014"
        return ", ".join(f"{r}\\,({z:+.1f})" for r, z in zip(vals, zvals))

    for direction, d_label in [("inc", "increased"), ("dec", "decreased")]:
        g_dz = zs.get("gnm_delta", {})
        a_dz = zs.get("anm_delta", {})
        g_res = g_dz.get(f"top5_{direction}_res", r1g.get(f"top5_{d_label}", []))
        g_z   = g_dz.get(f"top5_{direction}_z", [])
        a_res = a_dz.get(f"top5_{direction}_res", r1a.get(f"top5_{d_label}", []))
        a_z   = a_dz.get(f"top5_{direction}_z", [])
        if g_z:
            tex_lines.append(
                rf"GNM top 5 {d_label} & \multicolumn{{2}}{{r}}{{{_res_z_list_delta(g_res, g_z)}}} \\"
            )
        else:
            tex_lines.append(
                rf"GNM top 5 {d_label} & {{{_lst(r1g.get(f'top5_{d_label}', []))}}} & {{{_lst(r1a.get(f'top5_{d_label}', []))}}} \\"
            )
        if a_z:
            tex_lines.append(
                rf"ANM top 5 {d_label} & \multicolumn{{2}}{{r}}{{{_res_z_list_delta(a_res, a_z)}}} \\"
            )

    tex_lines += [
        r"\midrule",
        r"",
    ]

    # ── Section: Cross-Correlation (pattern 2) ──
    tex_lines += [
        r"\secrow{2. Cross-Correlation Comparison}",
        r"\midrule",
        rf"$\|\Delta CC\|_F$ (Frobenius norm) & {_f(r2g.get('delta_cc_frobenius_norm'))} & {_f(r2a.get('delta_cc_frobenius_norm'))} \\",
        rf"Mean $|\Delta CC|$ & {_f(r2g.get('delta_cc_abs_mean'))} & {_f(r2a.get('delta_cc_abs_mean'))} \\",
        rf"Max $|\Delta CC|$ & {_f(r2g.get('delta_cc_abs_max'))} & {_f(r2a.get('delta_cc_abs_max'))} \\",
        rf"Mean $|\Delta CC|$ at site {mutation_pos} & {_f(r2g.get('mean_abs_delta_at_site'))} & {_f(r2a.get('mean_abs_delta_at_site'))} \\",
        rf"Top 5 coupling-changed (residues) & {{{_lst(r2g.get('top5_coupling_changed', []))}}} & {{{_lst(r2a.get('top5_coupling_changed', []))}}} \\",
        r"\midrule",
        r"",
    ]

    # ── Section: Eigenvector Overlap (pattern 3) ──
    tex_lines += [
        r"\secrow{3. Eigenvector Overlap}",
        r"\midrule",
        rf"Mean diagonal overlap & {_f(r3g.get('mean_diagonal_overlap'))} & {_f(r3a.get('mean_diagonal_overlap'))} \\",
        rf"Min diagonal overlap & {_f(r3g.get('min_diagonal_overlap'))} & {_f(r3a.get('min_diagonal_overlap'))} \\",
        rf"Mode with min overlap & {r3g.get('min_overlap_mode', '—')} & {r3a.get('min_overlap_mode', '—')} \\",
        rf"RMSIP (first 10 modes) & {_f(r3g.get('rmsip_10'))} & {_f(r3a.get('rmsip_10'))} \\",
        rf"Mean mode overlap (ENM comparison) & {_f(comp_gnm.get('mean_mode_overlap'))} & {_f(comp_anm.get('mean_mode_overlap'))} \\",
    ]

    # Diagonal overlap per mode
    gnm_diag = r3g.get("diagonal_overlaps", [])
    anm_diag = r3a.get("diagonal_overlaps", [])
    n_show = min(10, max(len(gnm_diag), len(anm_diag)))
    for i in range(n_show):
        gv = gnm_diag[i] if i < len(gnm_diag) else None
        av = anm_diag[i] if i < len(anm_diag) else None
        tex_lines.append(
            rf"\quad Mode {i+1} overlap & {_f(gv)} & {_f(av)} \\"
        )

    tex_lines += [
        r"\midrule",
        r"",
    ]

    # ── Section: PRS / Allosteric (pattern 5) — before hinges for flow ──
    tex_lines += [
        r"\secrow{5. PRS / Allosteric Communication}",
        r"\midrule",
        rf"WT mean effectiveness & \multicolumn{{2}}{{c}}{{{_f(r5_wt.get('effectiveness_mean'))}}} \\",
        rf"WT mean sensitivity & \multicolumn{{2}}{{c}}{{{_f(r5_wt.get('sensitivity_mean'))}}} \\",
        rf"MUT mean effectiveness & \multicolumn{{2}}{{c}}{{{_f(r5_mut.get('effectiveness_mean'))}}} \\",
        rf"MUT mean sensitivity & \multicolumn{{2}}{{c}}{{{_f(r5_mut.get('sensitivity_mean'))}}} \\",
        rf"$\Delta$ effectiveness at site {mutation_pos} & \multicolumn{{2}}{{c}}{{{_sf(r5_dprs.get('delta_effectiveness_at_site'))}}} \\",
        rf"$\Delta$ sensitivity at site {mutation_pos} & \multicolumn{{2}}{{c}}{{{_sf(r5_dprs.get('delta_sensitivity_at_site'))}}} \\",
        rf"$\Delta$ effectiveness mean & \multicolumn{{2}}{{c}}{{{_sf(r5_dprs.get('delta_eff_mean'))}}} \\",
        rf"$\Delta$ sensitivity mean & \multicolumn{{2}}{{c}}{{{_sf(r5_dprs.get('delta_sen_mean'))}}} \\",
        rf"Top 5 effectors (WT) & \multicolumn{{2}}{{c}}{{{_lst(r5_wt.get('top5_effectors', []))}}} \\",
        rf"Top 5 sensors (WT) & \multicolumn{{2}}{{c}}{{{_lst(r5_wt.get('top5_sensors', []))}}} \\",
        rf"Top 5 effectors (MUT) & \multicolumn{{2}}{{c}}{{{_lst(r5_mut.get('top5_effectors', []))}}} \\",
        rf"Top 5 sensors (MUT) & \multicolumn{{2}}{{c}}{{{_lst(r5_mut.get('top5_sensors', []))}}} \\",
        rf"Top 5 $\Delta$eff increase & \multicolumn{{2}}{{c}}{{{_lst(r5_dprs.get('top5_eff_increase', []))}}} \\",
        rf"Top 5 $\Delta$sen increase & \multicolumn{{2}}{{c}}{{{_lst(r5_dprs.get('top5_sen_increase', []))}}} \\",
        r"\midrule",
        rf"N-term source residue & \multicolumn{{2}}{{c}}{{{r5_nterm.get('source_residue', '—')}}} \\",
        rf"WT mean N-term response & \multicolumn{{2}}{{c}}{{{_f(r5_nterm.get('wt_mean_response'))}}} \\",
        rf"MUT mean N-term response & \multicolumn{{2}}{{c}}{{{_f(r5_nterm.get('mut_mean_response'))}}} \\",
        rf"$\Delta$ N-term response mean & \multicolumn{{2}}{{c}}{{{_sf(r5_nterm.get('delta_mean'))}}} \\",
        rf"$\Delta$ N-term response max & \multicolumn{{2}}{{c}}{{{_sf(r5_nterm.get('delta_max'))}}} \\",
        rf"$\Delta$ N-term response min & \multicolumn{{2}}{{c}}{{{_sf(r5_nterm.get('delta_min'))}}} \\",
        rf"Top 5 N-term sens.\ increased & \multicolumn{{2}}{{c}}{{{_lst(r5_nterm.get('top5_increased_sensitivity', []))}}} \\",
        rf"Top 5 N-term sens.\ decreased & \multicolumn{{2}}{{c}}{{{_lst(r5_nterm.get('top5_decreased_sensitivity', []))}}} \\",
        r"",
    ]

    tex_lines += [
        r"\end{longtable}",
        r"}% end \small",
        r"",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 2 — Hinge-shift analysis (separate 4-col table)
    # ══════════════════════════════════════════════════════════════════════
    tex_lines += [
        r"% ────────────────────────────────────────────────────────────",
        r"% Table 2: Hinge-Shift Analysis",
        r"% ────────────────────────────────────────────────────────────",
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Hinge-shift analysis across the first "
        + str(n_hinge_modes) + r" GNM modes.}",
        r"\begin{tabular}{l r r r p{5cm}}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{Conserved} & \textbf{Lost} & \textbf{Gained} & \textbf{Conserved Residues} \\",
        r"\midrule",
    ]
    tex_lines += hinge_rows
    tex_lines += [
        r"\midrule",
        rf"Total & {total_conserved} & {total_lost} & {total_gained} & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 3 — Comprehensive per-mode analysis (GNM then ANM)
    # Merges eigenvalues, global variance %, site variance %, sqfluct at
    # site, Δsqfluct at site, cumulative %, and mode overlap into one
    # table per model.
    # ══════════════════════════════════════════════════════════════════════
    gnm_wt_eig = gnm_wt.get("eigenvalues", [])
    gnm_mut_eig = gnm_mut.get("eigenvalues", [])
    anm_wt_eig = anm_wt.get("eigenvalues", [])
    anm_mut_eig = anm_mut.get("eigenvalues", [])

    md_gw = mode_decomp.get("gnm_wt", {})
    md_gm = mode_decomp.get("gnm_mut", {})
    md_aw = mode_decomp.get("anm_wt", {})
    md_am = mode_decomp.get("anm_mut", {})

    # Mode overlaps (from pattern analysis)
    gnm_diag = r3g.get("diagonal_overlaps", [])
    anm_diag = r3a.get("diagonal_overlaps", [])

    def _mp(lst, i, fmt=".1f"):
        return f"{lst[i]:{fmt}}" if i < len(lst) else "—"

    for model_label, md_wt, md_mut, eig_wt, eig_mut, diag_ov in [
        ("GNM", md_gw, md_gm, gnm_wt_eig, gnm_mut_eig, gnm_diag),
        ("ANM", md_aw, md_am, anm_wt_eig, anm_mut_eig, anm_diag),
    ]:
        n_show = min(10, md_wt.get("n_modes", 0))
        if n_show == 0:
            continue

        pct_g_wt = md_wt.get("pct_global", [])
        pct_g_mt = md_mut.get("pct_global", [])
        cum_wt = md_wt.get("cum_global", [])
        cum_mt = md_mut.get("cum_global", [])
        pct_s_wt = md_wt.get("pct_site", [])
        pct_s_mt = md_mut.get("pct_site", [])
        sf_s_wt = md_wt.get("sf_site_per_mode", [])
        sf_s_mt = md_mut.get("sf_site_per_mode", [])

        # Determine sqfluct format based on magnitude
        is_anm = (model_label == "ANM")
        sf_fmt = ".4f" if not is_anm else ".2f"

        tex_lines += [
            r"% ────────────────────────────────────────────────────────────",
            rf"% Table 3{model_label[0]}: {model_label} per-mode analysis",
            r"% ────────────────────────────────────────────────────────────",
            r"\begin{table}[ht]",
            r"\centering\footnotesize",
            r"\caption{" + model_label
            + r" per-mode analysis (first " + str(n_show) + r" modes). "
            + r"$\lambda$ = eigenvalue; "
            + r"Glob.\,\% = fraction of total variance; "
            + r"Cum.\,\% = cumulative global variance; "
            + r"Site\,\% = fraction of variance at residue " + str(mutation_pos) + r"; "
            + r"SqFl = squared fluctuation at site; "
            + r"$\Delta$SqFl = MUT $-$ WT change at site; "
            + r"Ov = WT$\leftrightarrow$MUT eigenvector overlap.}",
            r"\begin{tabular}{l"
            + r" r@{\hskip 4pt}r"     # eigenvalues WT/MUT
            + r" r@{\hskip 4pt}r"     # global % WT/MUT
            + r" r"                     # cumulative %
            + r" r@{\hskip 4pt}r"     # site % WT/MUT
            + r" r@{\hskip 4pt}r"     # sqfluct at site WT/MUT
            + r" r"                     # Δsqfluct at site
            + r" r}",                   # overlap
            r"\toprule",
            r"\textbf{Mode}"
            + r" & \multicolumn{2}{c}{$\boldsymbol{\lambda}$}"
            + r" & \multicolumn{2}{c}{\textbf{Glob.\,\%}}"
            + r" & \textbf{Cum.\,\%}"
            + r" & \multicolumn{2}{c}{\textbf{Site\,\%}}"
            + r" & \multicolumn{2}{c}{\textbf{SqFl @ " + str(mutation_pos) + r"}}"
            + r" & $\boldsymbol{\Delta}$\textbf{SqFl}"
            + r" & \textbf{Ov} \\",
            r" & \textbf{WT} & \textbf{MUT}"
            + r" & \textbf{WT} & \textbf{MUT}"
            + r" & \textbf{WT}"
            + r" & \textbf{WT} & \textbf{MUT}"
            + r" & \textbf{WT} & \textbf{MUT}"
            + r" &"
            + r" & \\",
            r"\midrule",
        ]

        for i in range(n_show):
            ew = eig_wt[i] if i < len(eig_wt) else None
            em = eig_mut[i] if i < len(eig_mut) else None
            sw = sf_s_wt[i] if i < len(sf_s_wt) else None
            sm = sf_s_mt[i] if i < len(sf_s_mt) else None
            ds = (sm - sw) if (sw is not None and sm is not None) else None
            ov = diag_ov[i] if i < len(diag_ov) else None

            tex_lines.append(
                f"        {i+1}"
                f" & {_f(ew, '.6f')} & {_f(em, '.6f')}"
                f" & {_mp(pct_g_wt, i)} & {_mp(pct_g_mt, i)}"
                f" & {_mp(cum_wt, i)}"
                f" & {_mp(pct_s_wt, i)} & {_mp(pct_s_mt, i)}"
                f" & {_f(sw, sf_fmt)} & {_f(sm, sf_fmt)}"
                f" & {_sf(ds, '+' + sf_fmt)}"
                f" & {_f(ov)} \\\\"
            )

        # Cumulative / dominant summary
        tex_lines += [
            r"\midrule",
            f"        Cum.\ 1--{n_show}"
            f" & & & {_mp(pct_g_wt[:n_show] and cum_wt, n_show-1)}"
            f" & {_mp(cum_mt, n_show-1)}"
            f" & {_mp(cum_wt, n_show-1)}"
            f" & & & & & & \\\\",
        ]

        # Dominant mode callout
        dm_wt = md_wt.get("dominant_mode_at_site", "—")
        dp_wt = md_wt.get("dominant_mode_pct", 0)
        dm_mt = md_mut.get("dominant_mode_at_site", "—")
        dp_mt = md_mut.get("dominant_mode_pct", 0)
        tex_lines += [
            r"\midrule",
            rf"        \multicolumn{{12}}{{l}}{{\textit{{Dominant mode at site {mutation_pos}:}}"
            rf" WT = mode {dm_wt} ({dp_wt:.1f}\%), MUT = mode {dm_mt} ({dp_mt:.1f}\%)}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"",
        ]

    tex_lines += [
        r"\end{document}",
        r"",
    ]

    # ── Write file ───────────────────────────────────────────────────────
    tex_path = outdir / "summary_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))

    print(f"  LaTeX table: {tex_path}")
    return tex_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX summary table from pipeline results"
    )
    parser.add_argument("--analysis-dir", required=True,
                        help="Path to analysis/ directory")
    parser.add_argument("--pattern-dir", required=True,
                        help="Path to patterns/ directory")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for the .tex file")
    parser.add_argument("--rosetta-json", default=None,
                        help="Path to rosetta_results.json")
    parser.add_argument("--mutation-label", default="?",
                        help="Mutation label (e.g. R2103K)")
    parser.add_argument("--mutation-pos", type=int, default=0,
                        help="Mutation residue number")
    parser.add_argument("--modes", type=int, default=20,
                        help="Number of modes (default: 20)")
    args = parser.parse_args()

    generate_latex_table(
        analysis_dir=Path(args.analysis_dir),
        pattern_dir=Path(args.pattern_dir),
        outdir=Path(args.outdir),
        rosetta_json=Path(args.rosetta_json) if args.rosetta_json else None,
        mutation_label=args.mutation_label,
        mutation_pos=args.mutation_pos,
        n_modes=args.modes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
