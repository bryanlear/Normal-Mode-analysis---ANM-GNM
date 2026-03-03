#!/usr/bin/env python3
"""
Master pipeline: AlphaFold fetch → Rosetta mutation → GNM/ANM → Patterns → Plots

Workflow:
  1. Download wild-type structure from AlphaFold
  2. Introduce mutation via PyRosetta (restrained local FastRelax)
  3. Run GNM + ANM elastic-network-model analysis
  4. Run 5-part advanced pattern analysis
  5. Generate publication-grade figures
"""

import argparse
import json
import sys
import time
from pathlib import Path


def run_pipeline(
    uniprot_id: str = None,
    wt_pdb: Path = None,
    chain: str = "A",
    position: int = None,
    mutation: str = None,
    outdir: Path = Path("results"),
    fragment: int = 1,
    af_version: int = 6,
    protocol: str = "restrained-relax",
    repack_radius: float = 8.0,
    repack_rounds: int = 3,
    coord_cst_stdev: float = 0.5,
    n_modes: int = 20,
    skip_fetch: bool = False,
    skip_rosetta: bool = False,
    skip_enm: bool = False,
    skip_patterns: bool = False,
    skip_plots: bool = False,
    verbose: bool = False,
):
    """Run the full mutation analysis pipeline."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    pipeline_results = {}

    # ── Determine mutation label ──
    # We need the WT amino acid — resolve later from structure
    mutation_label = None  # will be set after we know WT residue

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Fetch structure from AlphaFold
    # ═══════════════════════════════════════════════════════════════════════
    if not skip_fetch and wt_pdb is None:
        print("\n" + "=" * 70)
        print("STEP 1: Downloading AlphaFold structure")
        print("=" * 70)

        if uniprot_id is None:
            print("ERROR: --uniprot required when not using --wt-pdb or --skip-fetch")
            return 1

        from fetch_structure import download_alphafold_structure
        wt_dir = outdir / "wt"
        wt_pdb = download_alphafold_structure(
            uniprot_id, wt_dir,
            fragment=fragment, version=af_version,
        )
        pipeline_results["wt_pdb"] = str(wt_pdb)
        print(f"  WT structure: {wt_pdb}")
    elif wt_pdb is not None:
        wt_pdb = Path(wt_pdb)
        if not wt_pdb.exists():
            print(f"ERROR: WT PDB not found: {wt_pdb}")
            return 1
        pipeline_results["wt_pdb"] = str(wt_pdb)
        print(f"\nUsing provided WT structure: {wt_pdb}")
    else:
        # skip_fetch + no wt_pdb: look for existing structure
        wt_dir = outdir / "wt"
        pdbs = list(wt_dir.glob("*.pdb"))
        if pdbs:
            wt_pdb = pdbs[0]
            pipeline_results["wt_pdb"] = str(wt_pdb)
            print(f"\nUsing cached WT structure: {wt_pdb}")
        else:
            print("ERROR: No WT PDB found. Use --uniprot or --wt-pdb")
            return 1

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Rosetta mutation
    # ═══════════════════════════════════════════════════════════════════════
    mut_pdb = outdir / "mutant" / f"{wt_pdb.stem}_{mutation}{position}.pdb"

    if not skip_rosetta:
        print("\n" + "=" * 70)
        print("STEP 2: Introducing mutation via PyRosetta")
        print("=" * 70)

        from mutate_structure import mutate_and_relax, normalize_aa
        mutation_aa = normalize_aa(mutation)

        rosetta_result = mutate_and_relax(
            pdb_path=wt_pdb,
            chain=chain,
            position=position,
            mutant_aa=mutation_aa,
            output_pdb=mut_pdb,
            protocol=protocol,
            repack_radius=repack_radius,
            repack_rounds=repack_rounds,
            coord_cst_stdev=coord_cst_stdev,
            verbose=verbose,
        )

        mutation_label = f"{rosetta_result['wildtype']}{position}{rosetta_result['mutant']}"
        pipeline_results["rosetta"] = rosetta_result
        pipeline_results["mutation_label"] = mutation_label

        print(f"\n  ΔΔG = {rosetta_result['ddg']:+.2f} REU ({rosetta_result['interpretation']})")
        print(f"  Mutant PDB: {mut_pdb}")

        # Save Rosetta results
        with open(outdir / "rosetta_results.json", "w") as f:
            json.dump(rosetta_result, f, indent=2)
    else:
        # Look for existing mutant PDB
        if not mut_pdb.exists():
            # Try to find any mutant PDB
            mut_dir = outdir / "mutant"
            if mut_dir.exists():
                pdbs = list(mut_dir.glob("*.pdb"))
                if pdbs:
                    mut_pdb = pdbs[0]
            if not mut_pdb.exists():
                print(f"ERROR: Mutant PDB not found: {mut_pdb}")
                return 1

        # Infer mutation label from filename if not set
        if mutation_label is None:
            from mutate_structure import normalize_aa
            mutation_aa = normalize_aa(mutation)
            mutation_label = f"?{position}{mutation_aa}"
            # Try to read WT residue from PDB
            try:
                import prody
                struct = prody.parsePDB(str(wt_pdb))
                ca = struct.select(f"calpha and resnum {position}")
                if ca is not None:
                    wt_aa = ca.getResnames()[0]
                    from mutate_structure import THREE_TO_ONE
                    wt_1 = THREE_TO_ONE.get(wt_aa, "?")
                    mutation_label = f"{wt_1}{position}{mutation_aa}"
            except Exception:
                pass

        pipeline_results["mutation_label"] = mutation_label
        print(f"\nUsing existing mutant structure: {mut_pdb}")

    pipeline_results["mut_pdb"] = str(mut_pdb)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: GNM + ANM analysis
    # ═══════════════════════════════════════════════════════════════════════
    enm_dir = outdir / "analysis"
    gnm_cutoff = None
    anm_cutoff = None

    if not skip_enm:
        print("\n" + "=" * 70)
        print("STEP 3: GNM + ANM Elastic-Network-Model Analysis")
        print("=" * 70)

        from enm_analysis import run_enm_analysis
        enm_result, gnm_wt, gnm_mut, anm_wt, anm_mut, resnums, gnm_cutoff, anm_cutoff = \
            run_enm_analysis(
                wt_pdb=wt_pdb,
                mut_pdb=mut_pdb,
                mutation_label=mutation_label,
                mutation_pos=position,
                out_dir=enm_dir,
                n_modes=n_modes,
            )
        pipeline_results["enm"] = {
            "gnm_cutoff": gnm_cutoff,
            "anm_cutoff": anm_cutoff,
            "gnm_mean_overlap": enm_result["comparison"]["gnm"]["mean_mode_overlap"],
            "anm_mean_overlap": enm_result["comparison"]["anm"]["mean_mode_overlap"],
        }
    else:
        # Try to load cutoffs from existing analysis
        mct_file = enm_dir / "mct_scan" / "mct_results.json"
        if mct_file.exists():
            with open(mct_file) as f:
                mct = json.load(f)
            gnm_cutoff = mct["gnm"]["cutoff"]
            anm_cutoff = mct["anm"]["cutoff"]
        print(f"\nSkipping ENM analysis (using existing data in {enm_dir})")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Pattern analysis
    # ═══════════════════════════════════════════════════════════════════════
    pattern_dir = outdir / "patterns"

    if not skip_patterns:
        print("\n" + "=" * 70)
        print("STEP 4: Advanced Pattern Analysis (5 components)")
        print("=" * 70)

        from pattern_analysis import run_pattern_analysis
        gc = gnm_cutoff if gnm_cutoff else 11.0
        ac = anm_cutoff if anm_cutoff else 11.0

        pattern_result = run_pattern_analysis(
            wt_pdb=wt_pdb,
            mut_pdb=mut_pdb,
            mutation_label=mutation_label,
            mutation_pos=position,
            out_dir=pattern_dir,
            gnm_cutoff=gc,
            anm_cutoff=ac,
            n_modes=n_modes,
        )
    else:
        print(f"\nSkipping pattern analysis (using existing data in {pattern_dir})")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Generate figures
    # ═══════════════════════════════════════════════════════════════════════
    fig_dir = outdir / "figures"

    if not skip_plots:
        print("\n" + "=" * 70)
        print("STEP 5: Generating Publication-Grade Figures")
        print("=" * 70)

        from plot_results import generate_all_plots
        generate_all_plots(
            data_dir=pattern_dir,
            fig_dir=fig_dir,
            mutation_label=mutation_label,
            mutation_pos=position,
            n_modes=n_modes,
        )
    else:
        print(f"\nSkipping plot generation")

    # ═══════════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    pipeline_results["elapsed_seconds"] = round(elapsed, 1)
    pipeline_results["output_directory"] = str(outdir)

    with open(outdir / "pipeline_results.json", "w") as f:
        json.dump(pipeline_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {mutation_label}")
    print(f"{'='*70}")
    print(f"  Output directory: {outdir}/")
    print(f"  WT structure:     {wt_pdb}")
    print(f"  MUT structure:    {mut_pdb}")
    if "rosetta" in pipeline_results:
        r = pipeline_results["rosetta"]
        print(f"  ΔΔG:              {r['ddg']:+.2f} REU ({r['interpretation']})")
    if "enm" in pipeline_results:
        e = pipeline_results["enm"]
        print(f"  GNM mean overlap: {e['gnm_mean_overlap']:.4f}")
        print(f"  ANM mean overlap: {e['anm_mean_overlap']:.4f}")
    print(f"  Total time:       {elapsed:.0f}s")
    print()
    print("  Output structure:")
    print(f"    {outdir}/")
    print(f"    ├── wt/              (wild-type PDB)")
    print(f"    ├── mutant/          (mutant PDB)")
    print(f"    ├── analysis/        (GNM + ANM raw data)")
    print(f"    │   ├── gnm_wt/     gnm_mut/")
    print(f"    │   ├── anm_wt/     anm_mut/")
    print(f"    │   ├── comparison/")
    print(f"    │   └── mct_scan/")
    print(f"    ├── patterns/        (5-part pattern analysis)")
    print(f"    │   ├── 1_msf_difference/")
    print(f"    │   ├── 2_crosscorr_comparison/")
    print(f"    │   ├── 3_eigenvector_overlap/")
    print(f"    │   ├── 4_hinge_shift/")
    print(f"    │   └── 5_prs_allosteric/")
    print(f"    ├── figures/         (PDF + PNG, 300 dpi)")
    print(f"    ├── pipeline_results.json")
    print(f"    └── rosetta_results.json")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Full mutation analysis pipeline: AlphaFold → Rosetta → GNM/ANM → Patterns → Plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from UniProt ID
  python run_pipeline.py --uniprot P08034 --chain A --position 13 --mutation M --outdir results/V13M

  # Use existing WT PDB
  python run_pipeline.py --wt-pdb protein.pdb --chain A --position 235 --mutation C --outdir results/F235C

  # Skip Rosetta (use pre-existing mutant PDB)
  python run_pipeline.py --wt-pdb wt.pdb --chain A --position 13 --mutation M --outdir results/V13M --skip-rosetta

  # Only regenerate plots
  python run_pipeline.py --wt-pdb wt.pdb --chain A --position 13 --mutation M --outdir results/V13M \\
         --skip-fetch --skip-rosetta --skip-enm --skip-patterns
        """
    )

    # ── Required ──
    parser.add_argument("--chain", required=True, help="Chain identifier (e.g. A)")
    parser.add_argument("--position", type=int, required=True, help="Residue position (PDB numbering)")
    parser.add_argument("--mutation", required=True,
                        help="Target amino acid (single or three-letter code, e.g. M or Met)")

    # ── Structure source (one required) ──
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--uniprot", help="UniProt accession to fetch from AlphaFold")
    src.add_argument("--wt-pdb", help="Path to existing wild-type PDB")

    # ── Output ──
    parser.add_argument("--outdir", default="results", help="Output directory (default: results/)")

    # ── AlphaFold options ──
    parser.add_argument("--fragment", type=int, default=1, help="AlphaFold fragment number (default: 1)")
    parser.add_argument("--af-version", type=int, default=4, help="AlphaFold model version (default: 4)")

    # ── Rosetta options ──
    parser.add_argument("--protocol", choices=["repack", "relax", "restrained-relax"],
                        default="restrained-relax",
                        help="Rosetta mutation protocol (default: restrained-relax)")
    parser.add_argument("--radius", type=float, default=8.0, help="Repack radius in Å (default: 8.0)")
    parser.add_argument("--rounds", type=int, default=3, help="Repack rounds (default: 3)")
    parser.add_argument("--coord-sdev", type=float, default=0.5,
                        help="Harmonic constraint SD for restrained-relax (default: 0.5)")

    # ── ENM options ──
    parser.add_argument("--modes", type=int, default=20, help="Number of normal modes (default: 20)")

    # ── Skip flags ──
    parser.add_argument("--skip-fetch", action="store_true", help="Skip AlphaFold download")
    parser.add_argument("--skip-rosetta", action="store_true", help="Skip Rosetta mutation step")
    parser.add_argument("--skip-enm", action="store_true", help="Skip ENM analysis")
    parser.add_argument("--skip-patterns", action="store_true", help="Skip pattern analysis")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")

    parser.add_argument("--verbose", action="store_true", help="Verbose PyRosetta output")

    args = parser.parse_args()

    return run_pipeline(
        uniprot_id=args.uniprot,
        wt_pdb=Path(args.wt_pdb) if args.wt_pdb else None,
        chain=args.chain,
        position=args.position,
        mutation=args.mutation,
        outdir=Path(args.outdir),
        fragment=args.fragment,
        af_version=args.af_version,
        protocol=args.protocol,
        repack_radius=args.radius,
        repack_rounds=args.rounds,
        coord_cst_stdev=args.coord_sdev,
        n_modes=args.modes,
        skip_fetch=args.skip_fetch,
        skip_rosetta=args.skip_rosetta,
        skip_enm=args.skip_enm,
        skip_patterns=args.skip_patterns,
        skip_plots=args.skip_plots,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
