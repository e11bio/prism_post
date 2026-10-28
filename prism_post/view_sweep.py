"""
Compare segmentation variants in neuroglancer.

Opens raw, enhanced, labels, the base segmentation and every variant of it:
segments/<affs>_relabeled*, and whatever the sweeps wrote to segments/sweep/
(06_sweep.py, and 07_barcode_relabel.py --sweep --write). Only raw and the
base segmentation start visible, toggle the others on and off to compare.

    uv run python prism_post/view_sweep.py
    uv run python prism_post/view_sweep.py --affs pred_affs
"""

import argparse

import neuroglancer

from prism_post.config import (
    COMBINED_AFFS,
    CROPS,
    DEFAULT_CROP,
    ENHANCED,
    LABELS,
    RAW,
    crop_path,
    default_data_dir,
    segments_name,
)
from prism_post.view import find_arrays, make_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affs", default=COMBINED_AFFS)
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--crop", type=int, choices=CROPS, default=DEFAULT_CROP)
    parser.add_argument("-b", "--bind-address", default="localhost")
    args = parser.parse_args()

    sample = crop_path(args.data_dir, args.crop)
    base = segments_name(args.affs)
    variants = [
        n
        for n in find_arrays(sample, skip=())
        if n.startswith(f"{base}_relabeled")
        or n.startswith(f"segments/sweep/{args.affs}_")
    ]
    names = [RAW, ENHANCED, LABELS, base] + sorted(variants)

    neuroglancer.set_server_bind_address(args.bind_address)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for name in names:
            s.layers[name] = make_layer(sample, name)
            s.layers[name].visible = name in (RAW, base)

    print(viewer)
    input("press enter to quit\n")
