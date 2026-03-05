#!/usr/bin/env python3
"""
Step 7 — Generate a comprehensive Markdown report (.md) with all pipeline
statistics organised into clearly labelled tables.

Reads:
  - analysis/master_results.json            (ENM cutoffs, eigenvalues, summaries)
  - analysis/comparison/comparison_summary.json (ΔSqFluct, mode overlaps, decomposition)
  - patterns/master_pattern_results.json    (MSF, cross-corr, overlap, hinges, PRS)
  - mode_explorer/exploration_summary.json  (per-mode rankings, % variance)
  - pipeline_results.json                   (overall pipeline metadata)
  - rosetta_results.json                    (Rosetta ΔΔG, if available)

Writes:
  - <outdir>/report.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(val, fmt=".4f"):
    """Format a float; return '—' for None/missing."""
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


def _pct(val, fmt=".2f"):
    """Format a percentage value."""
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}%"
    except (TypeError, ValueError):
        return str(val)


def _lst(vals, n=5, fmt=".0f"):
    """Comma-separated list of n values."""
    if not vals:
        return "—"
    return ", ".join(f"{v:{fmt}}" for v in vals[:n])


def _sci(val, digits=3):
    """Scientific notation."""
    if val is None:
        return "—"
    try:
        return f"{val:.{digits}e}"
    except (TypeError, ValueError):
        return str(val)


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dict."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def _md_table(headers, rows, alignments=None):
    """Build a Markdown table string.

    Parameters
    ----------
    headers : list[str]
    rows : list[list[str]]
    alignments : list[str] or None
        'l', 'c', or 'r' per column.
    """
    n = len(headers)
    if alignments is None:
        alignments = ["l"] + ["r"] * (n - 1)
    sep_map = {"l": ":---", "c": ":---:", "r": "---:"}

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep_map.get(a, "---") for a in alignments) + " |")
    for row in rows:
        # pad / truncate
        padded = list(row) + [""] * (n - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:n]) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Section builders — each returns a Markdown string
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(master, pipeline, rosetta, mutation_label, mutation_pos):
    """Title, timestamp, and pipeline overview."""
    lines = [
        f"# Elastic Network Model — Mutation Analysis Report",
        "",
        f"**Mutation:** {mutation_label}  ",
        f"**Position:** {mutation_pos}  ",
    ]
    wt = _safe_get(master, "pdb_wt") or _safe_get(pipeline, "wt_pdb") or "—"
    mt = _safe_get(master, "pdb_mut") or _safe_get(pipeline, "mut_pdb") or "—"
    lines += [
        f"**WT structure:** `{wt}`  ",
        f"**MUT structure:** `{mt}`  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
    ]
    elapsed = _safe_get(pipeline, "elapsed_seconds")
    if elapsed is not None:
        lines.append(f"**Pipeline runtime:** {elapsed:.0f} s  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _section_rosetta(rosetta):
    """Rosetta mutagenesis results (if available)."""
    if not rosetta:
        return ""
    lines = ["## 1. Rosetta Mutagenesis", ""]

    headers = ["Metric", "Value"]
    rows = [
        ["Protocol", str(_safe_get(rosetta, "protocol", default="—"))],
        ["WT score (REU)", _f(_safe_get(rosetta, "wt_score"), ".2f")],
        ["MUT score (REU)", _f(_safe_get(rosetta, "mut_score"), ".2f")],
        ["ΔΔG (REU)", _sf(_safe_get(rosetta, "ddg"), "+.2f")],
        ["Interpretation", str(_safe_get(rosetta, "interpretation", default="—"))],
    ]
    lines.append(_md_table(headers, rows, ["l", "r"]))
    lines += ["", ""]
    return "\n".join(lines)


def _section_cutoffs(master):
    """ENM cutoff / MCT scan summary."""
    lines = ["## 2. ENM Cutoff Selection (MCT Scan)", ""]

    for model in ("gnm", "anm"):
        c = _safe_get(master, "cutoffs", model)
        if not c:
            continue
        lines.append(f"### {model.upper()}")
        lines.append("")
        lines.append(f"**Selected cutoff:** {_f(c.get('selected'), '.1f')} Å")
        lines.append("")

        mct = c.get("mct_scan", [])
        if mct:
            headers = ["Cutoff (Å)", "Min contacts", "Mean contacts", "Disconnected"]
            rows = []
            for entry in mct:
                rows.append([
                    _f(entry["cutoff"], ".1f"),
                    str(entry["min_contacts"]),
                    _f(entry["mean_contacts"], ".2f"),
                    str(entry["disconnected"]),
                ])
            lines.append(_md_table(headers, rows, ["r", "r", "r", "r"]))
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def _section_enm_summary(master):
    """Per-model ENM summary (eigenvalues, sq fluctuations)."""
    lines = ["## 3. ENM Model Summary", ""]

    headers = ["Property", "GNM WT", "GNM MUT", "ANM WT", "ANM MUT"]
    gnm_wt = _safe_get(master, "gnm_wt_summary") or {}
    gnm_mt = _safe_get(master, "gnm_mut_summary") or {}
    anm_wt = _safe_get(master, "anm_wt_summary") or {}
    anm_mt = _safe_get(master, "anm_mut_summary") or {}

    rows = [
        ["Cutoff (Å)",
         _f(gnm_wt.get("cutoff_A"), ".1f"),
         _f(gnm_mt.get("cutoff_A"), ".1f"),
         _f(anm_wt.get("cutoff_A"), ".1f"),
         _f(anm_mt.get("cutoff_A"), ".1f")],
        ["Nᵣₑₛ",
         str(gnm_wt.get("n_residues", "—")),
         str(gnm_mt.get("n_residues", "—")),
         str(anm_wt.get("n_residues", "—")),
         str(anm_mt.get("n_residues", "—"))],
        ["Modes computed",
         str(gnm_wt.get("n_modes_computed", "—")),
         str(gnm_mt.get("n_modes_computed", "—")),
         str(anm_wt.get("n_modes_computed", "—")),
         str(anm_mt.get("n_modes_computed", "—"))],
        ["SqFluct mean",
         _f(gnm_wt.get("sq_flucts_mean")),
         _f(gnm_mt.get("sq_flucts_mean")),
         _f(anm_wt.get("sq_flucts_mean"), ".2f"),
         _f(anm_mt.get("sq_flucts_mean"), ".2f")],
        ["SqFluct std",
         _f(gnm_wt.get("sq_flucts_std")),
         _f(gnm_mt.get("sq_flucts_std")),
         _f(anm_wt.get("sq_flucts_std"), ".2f"),
         _f(anm_mt.get("sq_flucts_std"), ".2f")],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r", "r", "r"]))
    lines.append("")

    # Eigenvalue table
    lines.append("### Eigenvalues (modes 1–20)")
    lines.append("")
    eig_headers = ["Mode"] + [f"{m} {s}" for m in ("GNM", "ANM") for s in ("WT", "MUT")]
    eig_rows = []
    gnm_wt_eig = gnm_wt.get("eigenvalues", [])
    gnm_mt_eig = gnm_mt.get("eigenvalues", [])
    anm_wt_eig = anm_wt.get("eigenvalues", [])
    anm_mt_eig = anm_mt.get("eigenvalues", [])
    n = max(len(gnm_wt_eig), len(gnm_mt_eig), len(anm_wt_eig), len(anm_mt_eig))
    for i in range(n):
        eig_rows.append([
            str(i + 1),
            _sci(gnm_wt_eig[i]) if i < len(gnm_wt_eig) else "—",
            _sci(gnm_mt_eig[i]) if i < len(gnm_mt_eig) else "—",
            _sci(anm_wt_eig[i]) if i < len(anm_wt_eig) else "—",
            _sci(anm_mt_eig[i]) if i < len(anm_mt_eig) else "—",
        ])
    lines.append(_md_table(eig_headers, eig_rows, ["r"] * 5))
    lines += ["", ""]
    return "\n".join(lines)


def _section_comparison(comparison):
    """WT vs MUT comparison summary (ΔSqFluct, mode overlaps)."""
    if not comparison:
        return ""
    lines = ["## 4. WT vs MUT Comparison", ""]

    # Delta squared fluctuation stats
    lines.append("### Squared-Fluctuation Differences")
    lines.append("")
    headers = ["Metric", "GNM", "ANM"]
    gnm = _safe_get(comparison, "gnm") or {}
    anm = _safe_get(comparison, "anm") or {}
    rows = [
        ["Δ at mutation site", _sf(gnm.get("delta_sqflucts_at_site")), _sf(anm.get("delta_sqflucts_at_site"), "+.4f")],
        ["Δ mean (global)", _sf(gnm.get("delta_sqflucts_mean")), _sf(anm.get("delta_sqflucts_mean"), "+.4f")],
        ["Δ std", _f(gnm.get("delta_sqflucts_std")), _f(anm.get("delta_sqflucts_std"), ".4f")],
        ["Δ max", _sf(gnm.get("delta_sqflucts_max")), _sf(anm.get("delta_sqflucts_max"), "+.4f")],
        ["Δ min", _sf(gnm.get("delta_sqflucts_min")), _sf(anm.get("delta_sqflucts_min"), "+.4f")],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r"]))
    lines += ["", ""]

    # Mode overlap table
    lines.append("### Per-Mode Overlap (WT vs MUT eigenvectors)")
    lines.append("")
    gnm_overlaps = gnm.get("mode_overlaps_1_10", [])
    anm_overlaps = anm.get("mode_overlaps_1_10", [])
    n = max(len(gnm_overlaps), len(anm_overlaps))
    mo_headers = ["Mode", "GNM overlap", "ANM overlap"]
    mo_rows = []
    for i in range(n):
        mo_rows.append([
            str(i + 1),
            _f(gnm_overlaps[i], ".6f") if i < len(gnm_overlaps) else "—",
            _f(anm_overlaps[i], ".6f") if i < len(anm_overlaps) else "—",
        ])
    mo_rows.append([
        "**Mean**",
        f"**{_f(gnm.get('mean_mode_overlap'), '.6f')}**",
        f"**{_f(anm.get('mean_mode_overlap'), '.6f')}**",
    ])
    lines.append(_md_table(mo_headers, mo_rows, ["r", "r", "r"]))
    lines += ["", ""]

    return "\n".join(lines)


def _section_mode_decomposition(comparison):
    """Mode decomposition: % variance global and at mutation site."""
    decomp = _safe_get(comparison, "mode_decomposition")
    if not decomp:
        return ""
    lines = ["## 5. Mode Variance Decomposition", ""]

    for model_key in ("gnm", "anm"):
        md = decomp.get(model_key)
        if not md:
            continue
        wt = md.get("wt", {})
        mut = md.get("mut", {})

        lines.append(f"### {model_key.upper()}")
        lines.append("")
        lines.append(f"- **Dominant mode at site (WT):** Mode {wt.get('dominant_mode_at_site', '—')} "
                      f"({_pct(wt.get('dominant_mode_pct'))})")
        lines.append(f"- **Dominant mode at site (MUT):** Mode {mut.get('dominant_mode_at_site', '—')} "
                      f"({_pct(mut.get('dominant_mode_pct'))})")
        lines.append("")

        pct_g_wt = wt.get("pct_global", [])
        pct_g_mt = mut.get("pct_global", [])
        cum_g_wt = wt.get("cum_global", [])
        cum_g_mt = mut.get("cum_global", [])
        pct_s_wt = wt.get("pct_site", [])
        pct_s_mt = mut.get("pct_site", [])
        n_modes = max(len(pct_g_wt), len(pct_g_mt))

        headers = ["Mode", "% Global WT", "% Global MUT", "Cum % WT", "Cum % MUT",
                    "% Site WT", "% Site MUT"]
        rows = []
        for i in range(n_modes):
            rows.append([
                str(i + 1),
                _pct(pct_g_wt[i] if i < len(pct_g_wt) else None),
                _pct(pct_g_mt[i] if i < len(pct_g_mt) else None),
                _pct(cum_g_wt[i] if i < len(cum_g_wt) else None),
                _pct(cum_g_mt[i] if i < len(cum_g_mt) else None),
                _pct(pct_s_wt[i] if i < len(pct_s_wt) else None),
                _pct(pct_s_mt[i] if i < len(pct_s_mt) else None),
            ])
        lines.append(_md_table(headers, rows, ["r"] * 7))
        lines += ["", ""]

    return "\n".join(lines)


def _section_msf(patterns):
    """MSF difference analysis."""
    msf = _safe_get(patterns, "1_msf_difference")
    if not msf:
        return ""
    lines = ["## 6. Mean-Square Fluctuation (MSF) Differences", ""]

    headers = ["Metric", "GNM", "ANM"]
    gnm = msf.get("gnm", {})
    anm = msf.get("anm", {})
    rows = [
        ["MSF WT at site", _f(gnm.get("msf_wt_at_site")), _f(anm.get("msf_wt_at_site"), ".4f")],
        ["MSF MUT at site", _f(gnm.get("msf_mut_at_site")), _f(anm.get("msf_mut_at_site"), ".4f")],
        ["Δ at site", _sf(gnm.get("delta_at_site")), _sf(anm.get("delta_at_site"), "+.4f")],
        ["Fractional Δ at site", _f(gnm.get("frac_at_site")), _f(anm.get("frac_at_site"))],
        ["Δ mean (global)", _sf(gnm.get("delta_mean")), _sf(anm.get("delta_mean"), "+.4f")],
        ["Δ std", _f(gnm.get("delta_std")), _f(anm.get("delta_std"), ".4f")],
        ["Δ max", _sf(gnm.get("delta_max")), _sf(anm.get("delta_max"), "+.4f")],
        ["Δ min", _sf(gnm.get("delta_min")), _sf(anm.get("delta_min"), "+.4f")],
        ["Top-5 increased", _lst(gnm.get("top5_increased")), _lst(anm.get("top5_increased"))],
        ["Top-5 decreased", _lst(gnm.get("top5_decreased")), _lst(anm.get("top5_decreased"))],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r"]))
    lines += ["", ""]
    return "\n".join(lines)


def _section_crosscorr(patterns):
    """Cross-correlation comparison."""
    cc = _safe_get(patterns, "2_crosscorr_comparison")
    if not cc:
        return ""
    lines = ["## 7. ΔCC — Dynamic Cross-Correlation Matrix Analysis", ""]

    lines.append(
        "> The dynamic cross-correlation matrix CC\u1D62\u2C7C measures the "
        "time-averaged correlation of motions between every pair of residues "
        "(i, j).  ΔCC = CC\u1D39\u1D41\u1D40 − CC\u1D42\u1D40 quantifies how the "
        "mutation reshapes inter-residue coupling."
    )
    lines += ["", ""]

    # ── Global metrics ──
    lines.append("### 7.1 Global ΔCC Statistics")
    lines.append("")
    headers = ["Metric", "GNM", "ANM"]
    gnm = cc.get("gnm", {})
    anm = cc.get("anm", {})
    rows = [
        ["ΔCC Frobenius norm", _f(gnm.get("delta_cc_frobenius_norm"), ".4f"),
         _f(anm.get("delta_cc_frobenius_norm"), ".4f")],
        ["|ΔCC| mean", _sci(gnm.get("delta_cc_abs_mean")),
         _sci(anm.get("delta_cc_abs_mean"))],
        ["|ΔCC| max", _f(gnm.get("delta_cc_abs_max")),
         _f(anm.get("delta_cc_abs_max"))],
        ["Mean |Δ| at site row", _f(gnm.get("mean_abs_delta_at_site")),
         _f(anm.get("mean_abs_delta_at_site"))],
        ["Top-5 coupling changed", _lst(gnm.get("top5_coupling_changed")),
         _lst(anm.get("top5_coupling_changed"))],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r"]))
    lines += ["", ""]

    # ── Local vs distal ΔCC ──
    has_local = gnm.get("local_delta_cc_mean") is not None
    if has_local:
        lines.append("### 7.2 Local vs Distal ΔCC (±20 residues around site)")
        lines.append("")
        ld_headers = ["Region", "GNM mean |ΔCC|", "GNM max |ΔCC|",
                      "ANM mean |ΔCC|", "ANM max |ΔCC|"]
        ld_rows = [
            ["Local (±20 res)",
             _sci(gnm.get("local_delta_cc_mean")),
             _f(gnm.get("local_delta_cc_max")),
             _sci(anm.get("local_delta_cc_mean")),
             _f(anm.get("local_delta_cc_max"))],
            ["Distal (rest)",
             _sci(gnm.get("distal_delta_cc_mean")),
             _f(gnm.get("distal_delta_cc_max")),
             _sci(anm.get("distal_delta_cc_mean")),
             _f(anm.get("distal_delta_cc_max"))],
        ]
        lines.append(_md_table(ld_headers, ld_rows, ["l", "r", "r", "r", "r"]))
        lines += ["", ""]

    # ── Significance fractions ──
    has_frac = gnm.get("frac_pairs_gt_0.01") is not None
    if has_frac:
        lines.append("### 7.3 Fraction of Residue Pairs Exceeding |ΔCC| Thresholds")
        lines.append("")
        fr_headers = ["Threshold", "GNM", "ANM"]
        fr_rows = [
            ["|ΔCC| > 0.01",
             _pct(100 * gnm.get("frac_pairs_gt_0.01", 0)),
             _pct(100 * anm.get("frac_pairs_gt_0.01", 0))],
            ["|ΔCC| > 0.02",
             _pct(100 * gnm.get("frac_pairs_gt_0.02", 0)),
             _pct(100 * anm.get("frac_pairs_gt_0.02", 0))],
            ["|ΔCC| > 0.05",
             _pct(100 * gnm.get("frac_pairs_gt_0.05", 0)),
             _pct(100 * anm.get("frac_pairs_gt_0.05", 0))],
        ]
        lines.append(_md_table(fr_headers, fr_rows, ["l", "r", "r"]))
        lines += ["", ""]

    # ── Top-10 residue pairs with largest |ΔCC| ──
    for model_key, model_label in [("gnm", "GNM"), ("anm", "ANM")]:
        top_pairs = _safe_get(cc, model_key, "top10_pairs")
        if not top_pairs:
            continue
        lines.append(f"### 7.4 {model_label} — Top-10 Residue Pairs by |ΔCC|")
        lines.append("")
        tp_headers = ["Rank", "Res i", "Res j", "ΔCC", "|ΔCC|"]
        tp_rows = []
        for rank, p in enumerate(top_pairs, 1):
            tp_rows.append([
                str(rank),
                str(p["res_i"]),
                str(p["res_j"]),
                _sf(p["delta_cc"], "+.6f"),
                _f(p["abs_delta_cc"], ".6f"),
            ])
        lines.append(_md_table(tp_headers, tp_rows, ["r", "r", "r", "r", "r"]))
        lines += ["", ""]

    # ── Per-mode CC decomposition ──
    for model_key, model_label in [("gnm", "GNM"), ("anm", "ANM")]:
        per_mode = _safe_get(cc, model_key, "per_mode_cc")
        if not per_mode:
            continue
        lines.append(f"### 7.5 {model_label} — Per-Mode Covariance Decomposition at Site")
        lines.append("")
        pm_headers = ["Mode", "|Cov| mean WT", "|Cov| mean MUT", "Cov self WT",
                       "Cov self MUT", "% Total WT", "% Total MUT", "Δ|Cov| mean"]
        pm_rows = []
        for mode_key in sorted(per_mode.keys(), key=lambda k: int(k.split("_")[1])):
            m = per_mode[mode_key]
            mode_num = mode_key.split("_")[1]
            pm_rows.append([
                mode_num,
                _sci(m.get("mean_abs_cov_wt")),
                _sci(m.get("mean_abs_cov_mut")),
                _sci(m.get("cov_self_wt")),
                _sci(m.get("cov_self_mut")),
                _pct(m.get("pct_cov_wt")),
                _pct(m.get("pct_cov_mut")),
                _sci(m.get("delta_mean_abs_cov")),
            ])
        lines.append(_md_table(pm_headers, pm_rows, ["r"] * 8))
        lines += ["", ""]

    return "\n".join(lines)


def _section_overlap(patterns):
    """Eigenvector overlap / RMSIP."""
    ov = _safe_get(patterns, "3_eigenvector_overlap")
    if not ov:
        return ""
    lines = ["## 8. Eigenvector Overlap & RMSIP", ""]

    headers = ["Metric", "GNM", "ANM"]
    gnm = ov.get("gnm", {})
    anm = ov.get("anm", {})
    rows = [
        ["Modes compared", str(gnm.get("n_modes", "—")), str(anm.get("n_modes", "—"))],
        ["Mean diagonal overlap", _f(gnm.get("mean_diagonal_overlap"), ".6f"),
         _f(anm.get("mean_diagonal_overlap"), ".6f")],
        ["Min diagonal overlap", _f(gnm.get("min_diagonal_overlap"), ".6f"),
         _f(anm.get("min_diagonal_overlap"), ".6f")],
        ["Min overlap mode", str(gnm.get("min_overlap_mode", "—")),
         str(anm.get("min_overlap_mode", "—"))],
        ["RMSIP (top 10)", _f(gnm.get("rmsip_10"), ".6f"),
         _f(anm.get("rmsip_10"), ".6f")],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r"]))
    lines += ["", ""]

    # Per-mode diagonal overlap
    gnm_diag = gnm.get("diagonal_overlaps", [])
    anm_diag = anm.get("diagonal_overlaps", [])
    n = max(len(gnm_diag), len(anm_diag))
    if n:
        lines.append("### Per-Mode Diagonal Overlap")
        lines.append("")
        do_headers = ["Mode", "GNM", "ANM"]
        do_rows = []
        for i in range(n):
            do_rows.append([
                str(i + 1),
                _f(gnm_diag[i], ".6f") if i < len(gnm_diag) else "—",
                _f(anm_diag[i], ".6f") if i < len(anm_diag) else "—",
            ])
        lines.append(_md_table(do_headers, do_rows, ["r", "r", "r"]))
        lines += ["", ""]

    return "\n".join(lines)


def _section_hinges(patterns):
    """Hinge-shift analysis per mode."""
    hs = _safe_get(patterns, "4_hinge_shift")
    if not hs:
        return ""
    lines = ["## 9. Hinge-Shift Analysis (GNM)", ""]
    lines.append(f"**Modes analysed:** {hs.get('n_hinge_modes_analysed', '—')}")
    lines.append("")

    per_mode = hs.get("per_mode", {})
    if per_mode:
        # Summary table
        lines.append("### 9.1 Summary")
        lines.append("")
        headers = ["Mode", "# WT hinges", "# MUT hinges", "Conserved",
                    "Lost", "Gained"]
        rows = []
        for mode_key in sorted(per_mode.keys(), key=lambda k: int(k.split("_")[1])):
            m = per_mode[mode_key]
            mode_num = mode_key.split("_")[1]
            rows.append([
                mode_num,
                str(len(m.get("wt_hinges", []))),
                str(len(m.get("mut_hinges", []))),
                str(m.get("n_conserved", "—")),
                str(m.get("n_lost", "—")),
                str(m.get("n_gained", "—")),
            ])
        lines.append(_md_table(headers, rows, ["r"] * 6))
        lines += ["", ""]

        # Per-mode detail with residue lists
        lines.append("### 9.2 Per-Mode Hinge Details")
        lines.append("")
        for mode_key in sorted(per_mode.keys(), key=lambda k: int(k.split("_")[1])):
            m = per_mode[mode_key]
            mode_num = mode_key.split("_")[1]
            lines.append(f"**Mode {mode_num}:**")
            lines.append("")
            det_headers = ["Category", "Residues"]
            det_rows = [
                ["WT hinges", _lst(m.get("wt_hinges", []), n=30)],
                ["MUT hinges", _lst(m.get("mut_hinges", []), n=30)],
                ["Conserved", _lst(m.get("conserved", []), n=30)],
                ["Lost in MUT", _lst(m.get("lost_in_mutant", []), n=30) or "—"],
                ["Gained in MUT", _lst(m.get("gained_in_mutant", []), n=30) or "—"],
            ]
            lines.append(_md_table(det_headers, det_rows, ["l", "l"]))
            lines += ["", ""]

    return "\n".join(lines)


def _section_prs(patterns, pattern_dir=None):
    """Perturbation Response Scanning (PRS)."""
    prs = _safe_get(patterns, "5_prs_allosteric")
    if not prs:
        return ""
    lines = ["## 10. Perturbation Response Scanning (PRS)", ""]

    lines.append(
        "> PRS measures how a unit perturbation at residue i propagates to "
        "all other residues j. **Effectiveness** = average response caused by "
        "perturbing residue i. **Sensitivity** = average response of residue j "
        "to all perturbations."
    )
    lines += ["", ""]

    # ── WT / MUT global PRS ────────────────────────────────────
    lines.append("### 10.1 Global PRS Summary")
    lines.append("")
    headers = ["Metric", "WT", "MUT", "Δ (MUT − WT)"]
    wt = prs.get("wt", {})
    mut = prs.get("mut", {})
    dprs = prs.get("delta_prs", {})
    wt_eff = wt.get("effectiveness_mean")
    mut_eff = mut.get("effectiveness_mean")
    wt_sen = wt.get("sensitivity_mean")
    mut_sen = mut.get("sensitivity_mean")
    d_eff = (mut_eff - wt_eff) if wt_eff is not None and mut_eff is not None else None
    d_sen = (mut_sen - wt_sen) if wt_sen is not None and mut_sen is not None else None
    rows = [
        ["Effectiveness mean", _f(wt_eff, ".4f"), _f(mut_eff, ".4f"), _sf(d_eff, "+.6f")],
        ["Sensitivity mean", _f(wt_sen, ".4f"), _f(mut_sen, ".4f"), _sf(d_sen, "+.6f")],
    ]
    lines.append(_md_table(headers, rows, ["l", "r", "r", "r"]))
    lines += ["", ""]

    # ── Top effectors & sensors ────────────────────────────────
    lines.append("### 10.2 Top-5 Effectors & Sensors")
    lines.append("")
    es_headers = ["Rank", "WT Effector", "MUT Effector", "WT Sensor", "MUT Sensor"]
    wt_eff_list = wt.get("top5_effectors", [])
    mut_eff_list = mut.get("top5_effectors", [])
    wt_sen_list = wt.get("top5_sensors", [])
    mut_sen_list = mut.get("top5_sensors", [])
    es_rows = []
    for r in range(5):
        es_rows.append([
            str(r + 1),
            str(wt_eff_list[r]) if r < len(wt_eff_list) else "—",
            str(mut_eff_list[r]) if r < len(mut_eff_list) else "—",
            str(wt_sen_list[r]) if r < len(wt_sen_list) else "—",
            str(mut_sen_list[r]) if r < len(mut_sen_list) else "—",
        ])
    lines.append(_md_table(es_headers, es_rows, ["r", "r", "r", "r", "r"]))
    lines += ["", ""]

    # ── Δ PRS at Mutation Site ────────────────────────────────
    if dprs:
        lines.append("### 10.3 Δ PRS at Mutation Site")
        lines.append("")
        dp_headers = ["Metric", "Value"]
        dp_rows = [
            ["Δ effectiveness at site", _sf(dprs.get("delta_effectiveness_at_site"), "+.4f")],
            ["Δ sensitivity at site", _sf(dprs.get("delta_sensitivity_at_site"), "+.4f")],
            ["Δ effectiveness global mean", _sf(dprs.get("delta_eff_mean"), "+.6f")],
            ["Δ sensitivity global mean", _sf(dprs.get("delta_sen_mean"), "+.6f")],
            ["Top-5 Δeff increase", _lst(dprs.get("top5_eff_increase"))],
            ["Top-5 Δsen increase", _lst(dprs.get("top5_sen_increase"))],
        ]
        lines.append(_md_table(dp_headers, dp_rows, ["l", "r"]))
        lines += ["", ""]

    # ── N-terminal Propagation ────────────────────────────────
    nterm = prs.get("nterm_propagation")
    if nterm:
        lines.append("### 10.4 N-terminal Propagation")
        lines.append("")
        nt_headers = ["Metric", "Value"]
        nt_rows = [
            ["Source residue", str(nterm.get("source_residue", "—"))],
            ["WT mean response", _f(nterm.get("wt_mean_response"), ".4f")],
            ["MUT mean response", _f(nterm.get("mut_mean_response"), ".4f")],
            ["Δ mean response", _sf(nterm.get("delta_mean"), "+.4f")],
            ["Δ max response", _sf(nterm.get("delta_max"), "+.4f")],
            ["Δ min response", _sf(nterm.get("delta_min"), "+.4f")],
            ["Top-5 increased sensitivity", _lst(nterm.get("top5_increased_sensitivity"))],
            ["Top-5 decreased sensitivity", _lst(nterm.get("top5_decreased_sensitivity"))],
        ]
        lines.append(_md_table(nt_headers, nt_rows, ["l", "r"]))
        lines += ["", ""]

    return "\n".join(lines)


def _section_mode_explorer(explorer):
    """Mode exploration summary (rankings, % variance)."""
    if not explorer:
        return ""
    lines = ["## 11. Mode Explorer — Deep Per-Mode Analysis", ""]

    for model_key in ("gnm", "anm"):
        md = explorer.get(model_key)
        if not md:
            continue
        lines.append(f"### {model_key.upper()}")
        lines.append("")
        lines.append(f"- **Modes analysed:** {md.get('n_modes_analyzed', '—')} "
                      f"({_lst(md.get('modes_analyzed', []), n=20)})")
        lines.append(f"- **Top site modes:** {_lst(md.get('top_site_modes', []), n=5)}")
        lines.append(f"- **Top global modes:** {_lst(md.get('top_global_modes', []), n=5)}")
        lines.append("")

        # Variance table
        pct_g_wt = md.get("pct_global_wt", [])
        pct_g_mt = md.get("pct_global_mut", [])
        pct_s_wt = md.get("pct_site_wt", [])
        pct_s_mt = md.get("pct_site_mut", [])
        n = max(len(pct_g_wt), len(pct_g_mt), len(pct_s_wt), len(pct_s_mt))
        if n:
            v_headers = ["Mode", "% Global WT", "% Global MUT", "% Site WT", "% Site MUT"]
            v_rows = []
            for i in range(n):
                v_rows.append([
                    str(i + 1),
                    _pct(pct_g_wt[i] if i < len(pct_g_wt) else None),
                    _pct(pct_g_mt[i] if i < len(pct_g_mt) else None),
                    _pct(pct_s_wt[i] if i < len(pct_s_wt) else None),
                    _pct(pct_s_mt[i] if i < len(pct_s_mt) else None),
                ])
            lines.append(_md_table(v_headers, v_rows, ["r"] * 5))
            lines += ["", ""]

    return "\n".join(lines)


def _section_figures(outdir):
    """List all generated figure files."""
    fig_dir = outdir / "figures"
    explorer_dir = outdir / "mode_explorer"
    lines = ["## 12. Generated Figures", ""]

    for label, d in [("Standard figures", fig_dir), ("Mode explorer", explorer_dir)]:
        if not d.exists():
            continue
        pdfs = sorted(d.rglob("*.pdf"))
        pngs = sorted(d.rglob("*.png"))
        morphs = sorted(d.rglob("*.pdb"))
        if not pdfs and not pngs and not morphs:
            continue
        lines.append(f"### {label}")
        lines.append("")
        if pdfs:
            lines.append(f"**PDF files ({len(pdfs)}):**")
            lines.append("")
            for p in pdfs:
                lines.append(f"- `{p.relative_to(outdir)}`")
            lines.append("")
        if pngs:
            lines.append(f"**PNG files ({len(pngs)}):**")
            lines.append("")
            for p in pngs:
                lines.append(f"- `{p.relative_to(outdir)}`")
            lines.append("")
        if morphs:
            lines.append(f"**Morph PDB files ({len(morphs)}):**")
            lines.append("")
            for p in morphs:
                lines.append(f"- `{p.relative_to(outdir)}`")
            lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    analysis_dir: Path,
    pattern_dir: Path,
    outdir: Path,
    rosetta_json: Path = None,
    explorer_dir: Path = None,
    mutation_label: str = "—",
    mutation_pos: int = 0,
    n_modes: int = 20,
):
    """Generate the comprehensive Markdown report.

    Parameters
    ----------
    analysis_dir : Path   — directory with master_results.json + comparison/
    pattern_dir  : Path   — directory with master_pattern_results.json
    outdir       : Path   — top-level output directory (receives report.md)
    rosetta_json : Path   — rosetta_results.json (optional)
    explorer_dir : Path   — mode_explorer directory (optional)
    mutation_label : str
    mutation_pos : int
    n_modes : int
    """
    analysis_dir = Path(analysis_dir)
    pattern_dir = Path(pattern_dir)
    outdir = Path(outdir)

    # ── Load data ────────────────────────────────────────────────────────
    def _load(path):
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return {}

    master = _load(analysis_dir / "master_results.json")
    comparison = _load(analysis_dir / "comparison" / "comparison_summary.json")
    patterns = _load(pattern_dir / "master_pattern_results.json")
    pipeline = _load(outdir / "pipeline_results.json")
    rosetta = _load(rosetta_json) if rosetta_json else {}
    explorer = _load(explorer_dir / "exploration_summary.json") if explorer_dir else {}

    # ── Assemble report ──────────────────────────────────────────────────
    sections = [
        _section_header(master, pipeline, rosetta, mutation_label, mutation_pos),
        _section_rosetta(rosetta),
        _section_cutoffs(master),
        _section_enm_summary(master),
        _section_comparison(comparison),
        _section_mode_decomposition(comparison),
        _section_msf(patterns),
        _section_crosscorr(patterns),
        _section_overlap(patterns),
        _section_hinges(patterns),
        _section_prs(patterns),
        _section_mode_explorer(explorer),
        _section_figures(outdir),
    ]

    report = "\n".join(s for s in sections if s)

    # ── Write ─────────────────────────────────────────────────────────────
    report_path = outdir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  ✓ Report written to {report_path}  ({len(report):,} chars)")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Markdown report from pipeline results",
    )
    parser.add_argument("--analysis-dir", required=True,
                        help="Path to analysis/ directory")
    parser.add_argument("--pattern-dir", required=True,
                        help="Path to patterns/ directory")
    parser.add_argument("--outdir", required=True,
                        help="Top-level output directory (will receive report.md)")
    parser.add_argument("--rosetta-json", default=None,
                        help="Path to rosetta_results.json (optional)")
    parser.add_argument("--explorer-dir", default=None,
                        help="Path to mode_explorer/ directory (optional)")
    parser.add_argument("--mutation", default="—",
                        help="Mutation label (e.g. R2103K)")
    parser.add_argument("--position", type=int, default=0,
                        help="Mutation residue position")
    parser.add_argument("--modes", type=int, default=20,
                        help="Number of modes used")
    args = parser.parse_args()

    generate_report(
        analysis_dir=Path(args.analysis_dir),
        pattern_dir=Path(args.pattern_dir),
        outdir=Path(args.outdir),
        rosetta_json=Path(args.rosetta_json) if args.rosetta_json else None,
        explorer_dir=Path(args.explorer_dir) if args.explorer_dir else None,
        mutation_label=args.mutation,
        mutation_pos=args.position,
        n_modes=args.modes,
    )


if __name__ == "__main__":
    main()
