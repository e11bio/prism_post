"""
binary -> mask

Threshold the barcode expression into a 0/1 mask. Fragment extraction then
ignores anything outside it.

    uv run python prism_post/post/01_mask.py
"""

from volara.blockwise import Threshold
from volara.datasets import Labels, Raw
from volara.logging import set_log_basedir

from prism_post.blockwise import run
from prism_post.config import (
    BINARY,
    BLOCK_SIZE,
    MASK,
    MASK_THRESHOLD,
    argument_parser,
    crop_path,
)

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=8).parse_args()
    sample = crop_path(args.data_dir, args.crop)
    set_log_basedir(sample / "volara_logs")

    threshold = Threshold(
        # binary is stored as (1, z, y, x), channels=0 gives us (z, y, x)
        in_data=Raw(store=sample / BINARY, scale_shift=(1 / 255, 0), channels=0),
        mask=Labels(store=sample / MASK),
        threshold=MASK_THRESHOLD,
        block_size=BLOCK_SIZE,
        num_workers=args.num_workers,
    )
    run(threshold)
