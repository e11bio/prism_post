"""
pred_affs * pseudo_affs -> combined_affs

The predicted affinities know about boundaries, the pseudo affinities know
about barcodes. Multiplying them only keeps edges both agree on.

    uv run python prism_post/post/03_combine_affs.py
"""

from volara.datasets import Affs
from volara.logging import set_log_basedir

from prism_post.blockwise import run
from prism_post.config import (
    BLOCK_SIZE,
    COMBINED_AFFS,
    NEIGHBORHOOD,
    PRED_AFFS,
    PSEUDO_AFFS,
    affs_name,
    argument_parser,
    crop_path,
)
from prism_post.tasks import CombineAffs

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=8).parse_args()
    sample = crop_path(args.data_dir, args.crop)
    set_log_basedir(sample / "volara_logs")

    combine = CombineAffs(
        affs_a=Affs(store=sample / affs_name(PRED_AFFS), neighborhood=NEIGHBORHOOD),
        affs_b=Affs(store=sample / affs_name(PSEUDO_AFFS), neighborhood=NEIGHBORHOOD),
        out_affs=Affs(
            store=sample / affs_name(COMBINED_AFFS), neighborhood=NEIGHBORHOOD
        ),
        block_size=BLOCK_SIZE,
        num_workers=args.num_workers,
    )
    run(combine)
