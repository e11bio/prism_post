"""
raw -> enhanced

Cleans up each of the 18 barcode channels. All the other models use this as
input.

    uv run python prism_post/inference/01_enhanced.py
"""

from prism_post.config import ENHANCED, RAW, argument_parser
from prism_post.inference.predict import run_predict

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=1).parse_args()
    run_predict(
        "enhanced",
        RAW,
        [ENHANCED],
        args.data_dir,
        args.checkpoint_dir,
        num_workers=args.num_workers,
    )
