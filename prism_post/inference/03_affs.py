"""
enhanced -> pred_affs

9 affinity channels, 3 short range and 6 long range (see config.NEIGHBORHOOD).
The model also predicts lsds but we don't need them here.

    uv run python prism_post/inference/03_affs.py

To see what enhancement contributes, we can also predict straight from raw:

    uv run python prism_post/inference/03_affs.py --input raw
"""

from prism_post.config import (
    ENHANCED,
    PRED_AFFS,
    PRED_AFFS_RAW,
    RAW,
    argument_parser,
    crop_path,
)
from prism_post.inference.predict import run_predict

if __name__ == "__main__":
    parser = argument_parser(__doc__, workers=1)
    parser.add_argument(
        "--input",
        choices=["enhanced", "raw"],
        default="enhanced",
        help="input dataset (default: enhanced)",
    )
    args = parser.parse_args()

    in_dataset = ENHANCED if args.input == "enhanced" else RAW
    out = PRED_AFFS if args.input == "enhanced" else PRED_AFFS_RAW
    run_predict(
        "affs",
        in_dataset,
        [out, None],  # (affs, lsds) -> keep affs only
        crop_path(args.data_dir, args.crop),
        args.checkpoint_dir,
        num_workers=args.num_workers,
        affs=True,
    )
