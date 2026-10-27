"""
uniform -> pseudo_affs

Cosine similarity of the embedding along each affinity offset. High if two
voxels have the same barcode, no matter whether there is a boundary between
them.

    uv run python prism_post/post/02_pseudo_affs.py
"""

from volara.datasets import Affs, Raw
from volara.logging import set_log_basedir

from prism_post.blockwise import run
from prism_post.config import (
    BLOCK_SIZE,
    NEIGHBORHOOD,
    PSEUDO_AFFS,
    SAMPLE,
    UNIFORM,
    affs_name,
    argument_parser,
)
from prism_post.tasks import PseudoAffs

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=8).parse_args()
    sample = args.data_dir / SAMPLE
    set_log_basedir(args.data_dir / "volara_logs")

    pseudo_affs = PseudoAffs(
        # uint8 [0, 255] -> embedding range [-1, 1]
        embedding_data=Raw(store=sample / UNIFORM, scale_shift=(2 / 255, -1)),
        affs_data=Affs(
            store=sample / affs_name(PSEUDO_AFFS), neighborhood=NEIGHBORHOOD
        ),
        block_size=BLOCK_SIZE,
        num_workers=args.num_workers,
    )
    run(pseudo_affs)
