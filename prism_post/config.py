"""
Shared settings for all the scripts: where data and checkpoints live,
dataset names inside the crop zarr, the affinity neighborhood and the post
processing parameters.

Directories are picked in this order:

1. --data-dir / --checkpoint-dir on the command line
2. PRISM_POST_DATA_DIR / PRISM_POST_CHECKPOINT_DIR environment variables
3. ./data and ./checkpoints in the current directory
"""

import argparse
import os
from pathlib import Path

from funlib.geometry import Coordinate

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #

ENV_DATA_DIR = "PRISM_POST_DATA_DIR"
ENV_CHECKPOINT_DIR = "PRISM_POST_CHECKPOINT_DIR"


def default_data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA_DIR, Path.cwd() / "data"))


def default_checkpoint_dir() -> Path:
    return Path(os.environ.get(ENV_CHECKPOINT_DIR, Path.cwd() / "checkpoints"))


# the bucket has 14 labelled crops, but only crop_0 to crop_4 have the 18
# barcode channels the released models were trained on (the rest are from
# other imaging rounds with fewer channels). The tutorial uses one of them,
# pass --crop to the download and to any script for another
CROPS = [0, 1, 2, 3, 4]
DEFAULT_CROP = 2


def crop_name(crop: int) -> str:
    return f"crop_{crop}.zarr"


def crop_path(data_dir: Path, crop: int) -> Path:
    return Path(data_dir) / crop_name(crop)


def argument_parser(
    description: str, workers: int | None = None, crops: bool = False
) -> argparse.ArgumentParser:
    """
    Argument parser with the --data-dir / --checkpoint-dir / --crop options
    every script takes. Pass workers to also get a --num-workers option.
    crops=True lets --crop take several numbers.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help=f"where the crop zarrs live (default: ${ENV_DATA_DIR} or ./data)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(),
        help=f"where model checkpoints live (default: ${ENV_CHECKPOINT_DIR} or ./checkpoints)",
    )
    if crops:
        parser.add_argument(
            "--crop",
            type=int,
            nargs="+",
            choices=CROPS,
            default=[DEFAULT_CROP],
            help=f"which crops to download, e.g. --crop 2 3 (default: {DEFAULT_CROP})",
        )
    else:
        parser.add_argument(
            "--crop",
            type=int,
            choices=CROPS,
            default=DEFAULT_CROP,
            help=f"which crop to process, <data_dir>/crop_<n>.zarr (default: {DEFAULT_CROP})",
        )
    if workers is not None:
        parser.add_argument(
            "--num-workers",
            type=int,
            default=workers,
            help=f"number of parallel workers (default: {workers})",
        )
    return parser


# sample data and checkpoints

BUCKET = "e11bio-prism"
CROP_PREFIX = "ls/models/training_data/instance"  # crop_<n>.zarr under here
CHECKPOINT_KEY = "ls/models/checkpoints"

# the four models we need for segmentation (no synapses here)
MODEL_NAMES = ["enhanced", "affs", "uniform", "binary_channel_agnostic"]

# --------------------------------------------------------------------------- #
# Datasets inside <data_dir>/crop_<n>.zarr
# --------------------------------------------------------------------------- #
#
# crop_2.zarr/
#   raw, labels            downloaded
#   predictions/           network outputs + the mask
#   affs/                  all affinities, the main inputs to segmentation
#   fragments/<affs>       one per set of affinities we segment
#   dbs/<affs>.sqlite      fragment graphs
#   luts/<affs>.npz        fragment -> segment look up tables
#   segments/<affs>        segmentations
#   skeletons/<affs>.npz   segment skeletons, for the barcode relabel
#   segments/sweep/        what the sweeps write, see 06 and 07
#   *.csv, *.png           results of 05, 06 and 07 --sweep
#   volara_logs/           worker logs + block done cache, per task. These
#                          are what lets an interrupted task resume

RAW = "raw"  # 18 channel barcode data, uint8
LABELS = "labels"  # sparse instance ground truth, uint64

ENHANCED = "predictions/enhanced"  # raw -> enhanced barcodes
BINARY = "predictions/binary"  # enhanced -> barcode expression probability
UNIFORM = "predictions/uniform"  # enhanced -> 24d uniform embedding
MASK = "predictions/mask"  # thresholded BINARY

# affinities are referred to by their short name in the scripts, e.g.
# `04_segment.py --affs combined_affs`, and live under affs/
PRED_AFFS = "pred_affs"  # enhanced -> affinities
PRED_AFFS_RAW = "pred_affs_raw"  # raw -> affinities (for comparison only)
PSEUDO_AFFS = "pseudo_affs"  # cosine similarity of UNIFORM along NEIGHBORHOOD
COMBINED_AFFS = "combined_affs"  # PRED_AFFS * PSEUDO_AFFS


def affs_name(affs: str) -> str:
    return f"affs/{affs}"


def fragments_name(affs: str) -> str:
    return f"fragments/{affs}"


def segments_name(affs: str) -> str:
    return f"segments/{affs}"


def db_name(affs: str) -> str:
    return f"dbs/{affs}.sqlite"


def lut_name(affs: str) -> str:
    return f"luts/{affs}.npz"


def skeletons_name(affs: str) -> str:
    return f"skeletons/{affs}.npz"


# affinities

VOXEL_SIZE = Coordinate(400, 168, 168)  # nm, (z, y, x)

NEIGHBORHOOD = [
    Coordinate(1, 0, 0),
    Coordinate(0, 1, 0),
    Coordinate(0, 0, 1),
    Coordinate(6, 0, 0),
    Coordinate(0, 9, 0),
    Coordinate(0, 0, 9),
    Coordinate(12, 0, 0),
    Coordinate(0, 27, 0),
    Coordinate(0, 0, 27),
]
N_ADJ = 3  # the first N_ADJ offsets are direct neighbours, the rest long range

# post processing params

BLOCK_SIZE = Coordinate(160, 160, 160)
# context in voxels, needs to cover the largest affinity offset
CONTEXT = Coordinate(15, 40, 40)

# voxels with barcode expression above this go into the mask
MASK_THRESHOLD = 0.5

# mutex watershed biases for fragments, short/long biases, split perferring
FRAG_BIAS = [-0.4] * N_ADJ + [-0.9] * (len(NEIGHBORHOOD) - N_ADJ)

# only use some of the long range affinities, otherwise we split too much
STRIDES = [(1, 1, 1)] * 3 + [(3, 6, 6)] * 3 + [(6, 12, 12)] * 3
RANDOMIZED_STRIDES = True

# drop fragments smaller than this many voxels
REMOVE_DEBRIS = 64

# biases for the global mutex watershed on the fragment graph. split leaning
# on purpose: the barcode relabel (07) bridges gaps afterwards but can't undo
# a merge, so we want few false merges here without going all the way down
# to fragments. see 06_sweep.py for the tradeoff
SEG_BIAS = {"adj_weight": -0.4, "lr_weight": -0.9}
