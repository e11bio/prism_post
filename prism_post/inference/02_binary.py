"""
enhanced -> binary

Single channel, how likely is it that any barcode is expressed in this voxel.
We threshold it into a mask later so we don't extract fragments in background.

    uv run python prism_post/inference/02_binary.py
"""

from prism_post.config import BINARY, ENHANCED, argument_parser, crop_path
from prism_post.inference.predict import run_predict

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=1).parse_args()
    run_predict(
        "binary_channel_agnostic",
        ENHANCED,
        [BINARY],
        crop_path(args.data_dir, args.crop),
        args.checkpoint_dir,
        num_workers=args.num_workers,
    )
