#!/usr/bin/env bash
set -euo pipefail

# ── Resolve script directory (works with symlinks) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──
UNIPROT=""
WT_PDB=""
CHAIN=""
POSITION=""
MUTATION=""
OUTDIR=""
FRAGMENT=1
AF_VERSION=6
PROTOCOL="restrained-relax"
RADIUS=8.0
ROUNDS=3
COORD_SDEV=0.5
MODES=20
SKIP_FETCH=false
SKIP_ROSETTA=false
SKIP_ENM=false
SKIP_PATTERNS=false
SKIP_PLOTS=false
VERBOSE=false

# ── Color output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

usage() {
    cat <<EOF
${BOLD}Mutation Dynamics Analysis Pipeline${NC}

${CYAN}USAGE${NC}
    $(basename "$0") [OPTIONS]

${CYAN}REQUIRED${NC}
    --chain CHAIN           Chain identifier (e.g. A)
    --position POS          Residue position (PDB numbering)
    --mutation AA           Target amino acid (single-letter or three-letter)

${CYAN}STRUCTURE SOURCE (one required)${NC}
    --uniprot ID            UniProt accession (fetches from AlphaFold)
    --wt-pdb FILE           Path to existing wild-type PDB file

${CYAN}OUTPUT${NC}
    --outdir DIR            Output directory (default: results/<mutation>/)

${CYAN}ALPHAFOLD OPTIONS${NC}
    --fragment N            AlphaFold fragment number    (default: 1)
    --af-version N          AlphaFold model version      (default: 4)

${CYAN}ROSETTA OPTIONS${NC}
    --protocol PROTO        repack | relax | restrained-relax  (default: restrained-relax)
    --radius FLOAT          Repack neighbourhood radius Å     (default: 8.0)
    --rounds N              Independent repack trajectories    (default: 3)
    --coord-sdev FLOAT      Harmonic constraint SD Å           (default: 0.5)

${CYAN}ENM OPTIONS${NC}
    --modes N               Number of normal modes             (default: 20)

${CYAN}SKIP FLAGS${NC}
    --skip-fetch            Skip AlphaFold download
    --skip-rosetta          Skip Rosetta mutation (use existing mutant PDB)
    --skip-enm              Skip GNM + ANM analysis
    --skip-patterns         Skip 5-part pattern analysis
    --skip-plots            Skip figure generation
    --enm-only              Only run ENM + patterns + plots (skip fetch + rosetta)
    --plots-only            Only regenerate figures

${CYAN}GENERAL${NC}
    --verbose               Verbose PyRosetta output
    -h, --help              Show this help message

${CYAN}EXAMPLES${NC}
    # Full pipeline from UniProt accession
    $(basename "$0") --uniprot P08034 --chain A --position 13 --mutation M

    # Use existing PDB, custom output directory
    $(basename "$0") --wt-pdb AF-P08034-F1-model_v4.pdb --chain A \\
                     --position 235 --mutation C --outdir results/F235C

    # Only re-run analysis + plots (skip fetch and Rosetta)
    $(basename "$0") --wt-pdb wt.pdb --chain A --position 13 --mutation M \\
                     --enm-only --outdir results/V13M

    # Only regenerate plots from existing data
    $(basename "$0") --wt-pdb wt.pdb --chain A --position 13 --mutation M \\
                     --plots-only --outdir results/V13M

${CYAN}OUTPUT STRUCTURE${NC}
    <outdir>/
    ├── wt/                 Wild-type PDB
    ├── mutant/             Mutant PDB (from Rosetta)
    ├── analysis/           GNM + ANM raw data
    │   ├── gnm_wt/  gnm_mut/  anm_wt/  anm_mut/
    │   ├── comparison/
    │   └── mct_scan/
    ├── patterns/           5-part pattern analysis
    │   ├── 1_msf_difference/
    │   ├── 2_crosscorr_comparison/
    │   ├── 3_eigenvector_overlap/
    │   ├── 4_hinge_shift/
    │   └── 5_prs_allosteric/
    ├── figures/            Publication-grade plots (PDF + PNG, 300 dpi)
    ├── pipeline_results.json
    └── rosetta_results.json
EOF
    exit 0
}

die() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# ── Parse arguments ──
[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uniprot)      UNIPROT="$2";       shift 2 ;;
        --wt-pdb)       WT_PDB="$2";        shift 2 ;;
        --chain)        CHAIN="$2";         shift 2 ;;
        --position)     POSITION="$2";      shift 2 ;;
        --mutation)     MUTATION="$2";      shift 2 ;;
        --outdir)       OUTDIR="$2";        shift 2 ;;
        --fragment)     FRAGMENT="$2";      shift 2 ;;
        --af-version)   AF_VERSION="$2";    shift 2 ;;
        --protocol)     PROTOCOL="$2";      shift 2 ;;
        --radius)       RADIUS="$2";        shift 2 ;;
        --rounds)       ROUNDS="$2";        shift 2 ;;
        --coord-sdev)   COORD_SDEV="$2";    shift 2 ;;
        --modes)        MODES="$2";         shift 2 ;;
        --skip-fetch)   SKIP_FETCH=true;    shift ;;
        --skip-rosetta) SKIP_ROSETTA=true;  shift ;;
        --skip-enm)     SKIP_ENM=true;      shift ;;
        --skip-patterns) SKIP_PATTERNS=true; shift ;;
        --skip-plots)   SKIP_PLOTS=true;    shift ;;
        --enm-only)     SKIP_FETCH=true; SKIP_ROSETTA=true; shift ;;
        --plots-only)   SKIP_FETCH=true; SKIP_ROSETTA=true; SKIP_ENM=true; SKIP_PATTERNS=true; shift ;;
        --verbose)      VERBOSE=true;       shift ;;
        -h|--help)      usage ;;
        *)              die "Unknown option: $1" ;;
    esac
done

# ── Validate required arguments ──
[[ -z "$CHAIN" ]]    && die "Missing required argument: --chain"
[[ -z "$POSITION" ]] && die "Missing required argument: --position"
[[ -z "$MUTATION" ]] && die "Missing required argument: --mutation"

if [[ -z "$UNIPROT" && -z "$WT_PDB" ]]; then
    die "Must specify either --uniprot or --wt-pdb"
fi

if [[ -n "$WT_PDB" && ! -f "$WT_PDB" ]]; then
    die "WT PDB file not found: $WT_PDB"
fi

# ── Default output directory ──
if [[ -z "$OUTDIR" ]]; then
    # Create a sensible default from mutation info
    MUT_UPPER=$(echo "$MUTATION" | tr '[:lower:]' '[:upper:]')
    OUTDIR="results/${MUT_UPPER}${POSITION}"
fi

# ── Resolve Python interpreter (prefer conda env's python) ──
PYTHON="${PYTHON:-python}"

# ── Check Python dependencies ──
info "Checking Python dependencies ..."

"$PYTHON" -c "import numpy" 2>/dev/null || die "numpy not installed. Run: pip install numpy"
"$PYTHON" -c "import prody" 2>/dev/null || die "ProDy not installed. Run: pip install prody"
"$PYTHON" -c "import matplotlib" 2>/dev/null || die "matplotlib not installed. Run: pip install matplotlib"

if [[ "$SKIP_ROSETTA" == "false" ]]; then
    "$PYTHON" -c "import pyrosetta" 2>/dev/null || {
        warn "PyRosetta not installed. Rosetta step will be skipped."
        warn "Install via: pip install pyrosetta-installer && python -c 'import pyrosetta.distributed; pyrosetta.distributed.maybe_init()'"
        SKIP_ROSETTA=true
    }
fi

success "Dependencies OK"

# ── Print configuration ──
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Mutation Dynamics Analysis Pipeline${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Structure source:${NC}  ${UNIPROT:-$WT_PDB}"
echo -e "  ${CYAN}Chain:${NC}             $CHAIN"
echo -e "  ${CYAN}Position:${NC}          $POSITION"
echo -e "  ${CYAN}Mutation:${NC}          $MUTATION"
echo -e "  ${CYAN}Protocol:${NC}          $PROTOCOL"
echo -e "  ${CYAN}Output:${NC}            $OUTDIR/"
echo -e "  ${CYAN}Modes:${NC}             $MODES"
echo ""
[[ "$SKIP_FETCH"    == "true" ]] && echo -e "  ${YELLOW}SKIP:${NC} AlphaFold download"
[[ "$SKIP_ROSETTA"  == "true" ]] && echo -e "  ${YELLOW}SKIP:${NC} Rosetta mutation"
[[ "$SKIP_ENM"      == "true" ]] && echo -e "  ${YELLOW}SKIP:${NC} ENM analysis"
[[ "$SKIP_PATTERNS" == "true" ]] && echo -e "  ${YELLOW}SKIP:${NC} Pattern analysis"
[[ "$SKIP_PLOTS"    == "true" ]] && echo -e "  ${YELLOW}SKIP:${NC} Figure generation"
echo ""

# ── Build Python command ──
PYTHON_CMD=(
    "$PYTHON" "${SCRIPT_DIR}/run_pipeline.py"
    --chain "$CHAIN"
    --position "$POSITION"
    --mutation "$MUTATION"
    --outdir "$OUTDIR"
    --fragment "$FRAGMENT"
    --af-version "$AF_VERSION"
    --protocol "$PROTOCOL"
    --radius "$RADIUS"
    --rounds "$ROUNDS"
    --coord-sdev "$COORD_SDEV"
    --modes "$MODES"
)

if [[ -n "$UNIPROT" ]]; then
    PYTHON_CMD+=(--uniprot "$UNIPROT")
elif [[ -n "$WT_PDB" ]]; then
    PYTHON_CMD+=(--wt-pdb "$WT_PDB")
fi

[[ "$SKIP_FETCH"    == "true" ]] && PYTHON_CMD+=(--skip-fetch)
[[ "$SKIP_ROSETTA"  == "true" ]] && PYTHON_CMD+=(--skip-rosetta)
[[ "$SKIP_ENM"      == "true" ]] && PYTHON_CMD+=(--skip-enm)
[[ "$SKIP_PATTERNS" == "true" ]] && PYTHON_CMD+=(--skip-patterns)
[[ "$SKIP_PLOTS"    == "true" ]] && PYTHON_CMD+=(--skip-plots)
[[ "$VERBOSE"       == "true" ]] && PYTHON_CMD+=(--verbose)

# ── Run ──
START_TIME=$(date +%s)
info "Starting pipeline ..."
echo ""

"${PYTHON_CMD[@]}"
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS_REMAIN=$((ELAPSED % 60))

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    success "Pipeline completed successfully in ${MINUTES}m ${SECONDS_REMAIN}s"
    echo ""
    echo -e "  ${CYAN}Results:${NC}  $OUTDIR/"
    echo -e "  ${CYAN}Figures:${NC}  $OUTDIR/figures/"
    echo ""
else
    die "Pipeline failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
