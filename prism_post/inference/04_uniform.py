"""
enhanced -> uniform

A 24 channel embedding of the barcode at each voxel. Voxels with the same
barcode end up close together, different barcodes far apart. Values are in
[-1, 1] and stored as uint8.

    uv run python prism_post/inference/04_uniform.py
"""

from prism_post.config import ENHANCED, UNIFORM, argument_parser
from prism_post.inference.predict import run_predict

if __name__ == "__main__":
    args = argument_parser(__doc__, workers=1).parse_args()
    run_predict(
        "uniform",
        ENHANCED,
        [UNIFORM],
        args.data_dir,
        args.checkpoint_dir,
        num_workers=args.num_workers,
    )
