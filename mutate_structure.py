#!/usr/bin/env python3
"""
Introduce a point mutation into a PDB structure using PyRosetta.

Supports three protocols:
  1. Local repacking (fast)
  2. Local FastRelax (repack + minimization)
  3. Restrained local FastRelax (harmonic BB constraints — preserves
     global Cα geometry for downstream ANM/GNM analysis)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional


# ── Amino acid code conversion ──────────────────────────────────────────────
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
ONE_LETTER_AA = set(THREE_TO_ONE.values())


def normalize_aa(token: str) -> str:
    """Accept single-letter, three-letter, or full name → single-letter."""
    t = token.strip().upper()
    if len(t) == 1 and t in ONE_LETTER_AA:
        return t
    if len(t) == 3 and t in THREE_TO_ONE:
        return THREE_TO_ONE[t]
    raise ValueError(
        f"Unrecognised amino acid '{token}'. "
        f"Use single-letter (e.g. H) or three-letter (e.g. His)."
    )


def aa1_to_rosetta(one_letter: str) -> str:
    """Single-letter → Rosetta residue type name (three-letter)."""
    return ONE_TO_THREE[one_letter]


def _setup_pyrosetta(verbose: bool = False, coord_cst_stdev: float = 0.5):
    """Initialize PyRosetta (idempotent)."""
    import pyrosetta
    options = (
        "-ignore_unrecognized_res "
        "-ignore_zero_occupancy false "
        "-ex1 -ex2aro "
        "-no_his_his_pairE "
        "-detect_disulf false "
        "-missing_density_to_jump "
        f"-relax:coord_cst_stdev {coord_cst_stdev} "
    )
    if not verbose:
        options += "-mute all "
    pyrosetta.init(extra_options=options)


def _make_local_task_factory(pose_position: int, radius: float = 8.0):
    import pyrosetta
    from pyrosetta import rosetta
    tf = rosetta.core.pack.task.TaskFactory()
    tf.push_back(rosetta.core.pack.task.operation.IncludeCurrent())

    focus = rosetta.core.select.residue_selector.ResidueIndexSelector(pose_position)
    neighbourhood = rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    neighbourhood.set_focus_selector(focus)
    neighbourhood.set_distance(radius)
    neighbourhood.set_include_focus_in_subset(True)

    prevent = rosetta.core.pack.task.operation.OperateOnResidueSubset(
        rosetta.core.pack.task.operation.PreventRepackingRLT(),
        neighbourhood, flip_subset=True,
    )
    tf.push_back(prevent)

    restrict = rosetta.core.pack.task.operation.OperateOnResidueSubset(
        rosetta.core.pack.task.operation.RestrictToRepackingRLT(),
        neighbourhood, flip_subset=False,
    )
    tf.push_back(restrict)
    return tf


def _make_local_movemap(pose, pose_position: int, radius: float = 8.0):
    from pyrosetta import rosetta
    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)

    focus = rosetta.core.select.residue_selector.ResidueIndexSelector(pose_position)
    neighbourhood = rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    neighbourhood.set_focus_selector(focus)
    neighbourhood.set_distance(radius)
    neighbourhood.set_include_focus_in_subset(True)

    subset = neighbourhood.apply(pose)
    for i in range(1, pose.total_residue() + 1):
        if subset[i]:
            mm.set_bb(i, True)
            mm.set_chi(i, True)
    return mm


def _local_fastrelax(pose, pose_position, scorefxn, radius=8.0):
    from pyrosetta import rosetta
    work = pose.clone()
    tf = _make_local_task_factory(pose_position, radius)
    mm = _make_local_movemap(work, pose_position, radius)
    relax = rosetta.protocols.relax.FastRelax(scorefxn, 5)
    relax.set_task_factory(tf)
    relax.set_movemap(mm)
    relax.apply(work)
    return work


def _restrained_local_fastrelax(pose, pose_position, scorefxn, radius=8.0):
    from pyrosetta import rosetta
    work = pose.clone()
    tf = _make_local_task_factory(pose_position, radius)
    mm = _make_local_movemap(work, pose_position, radius)
    relax = rosetta.protocols.relax.FastRelax(scorefxn, 5)
    relax.set_task_factory(tf)
    relax.set_movemap(mm)
    relax.constrain_relax_to_start_coords(True)
    relax.apply(work)
    return work


def mutate_and_relax(
    pdb_path: Path,
    chain: str,
    position: int,
    mutant_aa: str,
    output_pdb: Path,
    protocol: str = "restrained-relax",
    repack_radius: float = 8.0,
    repack_rounds: int = 3,
    coord_cst_stdev: float = 0.5,
    verbose: bool = False,
) -> Dict:
    """Introduce a mutation and relax the structure.

    Parameters
    ----------
    pdb_path : Path
        Input PDB file (wild-type).
    chain : str
        Chain identifier.
    position : int
        PDB residue number.
    mutant_aa : str
        Target amino acid (single-letter).
    output_pdb : Path
        Where to save the mutant structure.
    protocol : str
        "repack", "relax", or "restrained-relax".
    repack_radius : float
        Neighbourhood radius in Angstroms.
    repack_rounds : int
        Number of independent trajectories.
    coord_cst_stdev : float
        Harmonic constraint SD (for restrained-relax).
    verbose : bool
        PyRosetta verbosity.

    Returns
    -------
    dict with ΔΔG, energies, interpretation.
    """
    import pyrosetta
    from pyrosetta import rosetta

    _setup_pyrosetta(verbose=verbose, coord_cst_stdev=coord_cst_stdev)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    print(f"  Loaded structure: {pose.total_residue()} residues")

    pose_position = pose.pdb_info().pdb2pose(chain, position)
    if pose_position == 0:
        raise ValueError(f"Position {chain}:{position} not found in structure")

    wildtype_aa = pose.residue(pose_position).name1()
    mutant_aa = normalize_aa(mutant_aa)

    print(f"  Mutation: {wildtype_aa}{position}{mutant_aa}")
    print(f"  Protocol: {protocol}")
    print(f"  Radius: {repack_radius} Å, Rounds: {repack_rounds}")

    if wildtype_aa == mutant_aa:
        # No mutation needed — just copy
        pose.dump_pdb(str(output_pdb))
        return {
            "wildtype": wildtype_aa, "mutant": mutant_aa,
            "position": position, "chain": chain,
            "ddg": 0.0, "wt_energy": 0.0, "mut_energy": 0.0,
            "interpretation": "Same amino acid (no mutation)",
            "protocol": protocol,
        }

    # Select score function and relax method
    if protocol == "restrained-relax":
        scorefxn = pyrosetta.create_score_function("ref2015_cst")
        relax_fn = _restrained_local_fastrelax
    elif protocol == "relax":
        scorefxn = pyrosetta.create_score_function("ref2015")
        relax_fn = _local_fastrelax
    else:  # repack
        scorefxn = pyrosetta.create_score_function("ref2015")
        relax_fn = None

    best_wt = float("inf")
    best_mut = float("inf")
    best_mut_pose = None

    for r in range(1, repack_rounds + 1):
        t0 = time.time()

        if relax_fn is not None:
            # WT relax
            wt_relaxed = relax_fn(pose, pose_position, scorefxn, repack_radius)
            wt_e = scorefxn(wt_relaxed)
            if wt_e < best_wt:
                best_wt = wt_e

            # Mutant: mutate then relax
            mut_pose = pose.clone()
            rosetta.protocols.simple_moves.MutateResidue(
                pose_position, aa1_to_rosetta(mutant_aa)
            ).apply(mut_pose)
            mut_relaxed = relax_fn(mut_pose, pose_position, scorefxn, repack_radius)
            mut_e = scorefxn(mut_relaxed)
            if mut_e < best_mut:
                best_mut = mut_e
                best_mut_pose = mut_relaxed
        else:
            # Repack-only protocol
            tf = _make_local_task_factory(pose_position, repack_radius)

            wt_pose = pose.clone()
            packer_wt = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
            packer_wt.task_factory(tf)
            packer_wt.apply(wt_pose)
            wt_e = scorefxn(wt_pose)
            if wt_e < best_wt:
                best_wt = wt_e

            mut_pose = pose.clone()
            rosetta.protocols.simple_moves.MutateResidue(
                pose_position, aa1_to_rosetta(mutant_aa)
            ).apply(mut_pose)
            packer_mut = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
            packer_mut.task_factory(tf)
            packer_mut.apply(mut_pose)
            mut_e = scorefxn(mut_pose)
            if mut_e < best_mut:
                best_mut = mut_e
                best_mut_pose = mut_pose

        elapsed = time.time() - t0
        print(f"  Round {r}/{repack_rounds}: WT={wt_e:.1f}  MUT={mut_e:.1f}  ({elapsed:.1f}s)")

    ddg = best_mut - best_wt

    if ddg > 1.0:
        interpretation = "Destabilizing (likely pathogenic)"
    elif ddg < -1.0:
        interpretation = "Stabilizing (likely benign)"
    else:
        interpretation = "Neutral"

    # Save best mutant structure
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    best_mut_pose.dump_pdb(str(output_pdb))
    print(f"  Saved mutant structure: {output_pdb}")

    return {
        "wildtype": wildtype_aa,
        "mutant": mutant_aa,
        "position": position,
        "chain": chain,
        "ddg": round(ddg, 4),
        "wt_energy": round(best_wt, 4),
        "mut_energy": round(best_mut, 4),
        "interpretation": interpretation,
        "protocol": protocol,
        "repack_radius": repack_radius,
        "repack_rounds": repack_rounds,
    }


def main():
    parser = argparse.ArgumentParser(description="Introduce mutation via PyRosetta")
    parser.add_argument("--pdb", required=True, help="Input PDB file (wild-type)")
    parser.add_argument("--chain", required=True, help="Chain identifier")
    parser.add_argument("--position", type=int, required=True, help="Residue position (PDB numbering)")
    parser.add_argument("--mutation", required=True, help="Target amino acid (single-letter or three-letter)")
    parser.add_argument("--output", required=True, help="Output PDB file for mutant")
    parser.add_argument("--protocol", choices=["repack", "relax", "restrained-relax"],
                        default="restrained-relax",
                        help="Mutation protocol (default: restrained-relax)")
    parser.add_argument("--radius", type=float, default=8.0, help="Repack radius (default: 8.0)")
    parser.add_argument("--rounds", type=int, default=3, help="Repack rounds (default: 3)")
    parser.add_argument("--coord-sdev", type=float, default=0.5, help="Constraint SD (default: 0.5)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = mutate_and_relax(
        pdb_path=Path(args.pdb),
        chain=args.chain,
        position=args.position,
        mutant_aa=args.mutation,
        output_pdb=Path(args.output),
        protocol=args.protocol,
        repack_radius=args.radius,
        repack_rounds=args.rounds,
        coord_cst_stdev=args.coord_sdev,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"ΔΔG = {result['ddg']:+.2f} REU  ({result['interpretation']})")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
