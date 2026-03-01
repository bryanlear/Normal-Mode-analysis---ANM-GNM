#!/usr/bin/env python3
"""
Download a protein structure from AlphaFold DB by UniProt accession.

Fetches the PDB-format model from https://alphafold.ebi.ac.uk/ and
caches it locally.  Supports specifying fragment number for large
multi-fragment proteins.
"""

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path


ALPHAFOLD_PDB_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F{fragment}-model_v{version}.pdb"
)
ALPHAFOLD_CIF_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F{fragment}-model_v{version}.cif"
)


def download_alphafold_structure(
    uniprot_id: str,
    out_dir: Path,
    fragment: int = 1,
    version: int = 4,
    fmt: str = "pdb",
    force: bool = False,
) -> Path:
    """Download an AlphaFold structure and return the local file path.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g. "P08034").
    out_dir : Path
        Directory to save the structure.
    fragment : int
        AlphaFold fragment number (default 1).
    version : int
        AlphaFold model version (default 4; tries 4 then falls back to 3/2).
    fmt : str
        "pdb" or "cif".
    force : bool
        Re-download even if cached.

    Returns
    -------
    Path to the downloaded file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accession = uniprot_id.upper().strip()
    ext = "pdb" if fmt == "pdb" else "cif"
    filename = f"AF-{accession}-F{fragment}-model_v{version}.{ext}"
    local_path = out_dir / filename

    if local_path.exists() and not force:
        print(f"  Cached: {local_path}")
        return local_path

    url_template = ALPHAFOLD_PDB_URL if fmt == "pdb" else ALPHAFOLD_CIF_URL

    # Try requested version, then fall back
    for v in [version, 4, 3, 2]:
        url = url_template.format(accession=accession, fragment=fragment, version=v)
        try:
            print(f"  Fetching {url} ...")
            urllib.request.urlretrieve(url, str(local_path))
            # Update filename if fallback version used
            if v != version:
                actual = f"AF-{accession}-F{fragment}-model_v{v}.{ext}"
                actual_path = out_dir / actual
                if actual_path != local_path:
                    local_path.rename(actual_path)
                    local_path = actual_path
            print(f"  Saved: {local_path}")
            return local_path
        except urllib.error.HTTPError as e:
            if e.code == 404 and v != 2:
                print(f"  Version {v} not found, trying v{v-1} ...")
                continue
            raise RuntimeError(
                f"Failed to download AlphaFold structure for {accession} "
                f"(fragment {fragment}): {e}"
            ) from e

    raise RuntimeError(
        f"No AlphaFold structure found for {accession} (fragment {fragment})"
    )


def main():
    parser = argparse.ArgumentParser(description="Download AlphaFold structure")
    parser.add_argument("--uniprot", required=True, help="UniProt accession ID")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--fragment", type=int, default=1, help="Fragment number (default: 1)")
    parser.add_argument("--version", type=int, default=4, help="Model version (default: 4)")
    parser.add_argument("--format", choices=["pdb", "cif"], default="pdb")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    path = download_alphafold_structure(
        args.uniprot,
        Path(args.outdir),
        fragment=args.fragment,
        version=args.version,
        fmt=args.format,
        force=args.force,
    )
    print(f"\nStructure: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
