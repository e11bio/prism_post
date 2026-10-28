"""
Score the segmentations against the labels.

For each segmentation we report variation of information (voi_split goes up
when one neuron is split into several segments, voi_merge when one segment
covers several neurons, lower is better), how much of the labelled tissue the
segmentation covers at all, and plain counts of false merges and false splits.
See prism_post/evaluate.py for the details, in short: unassigned voxels are
left out of VOI and reported as coverage instead, and a merge is a segment
with more than one label in it.

    uv run python prism_post/post/05_evaluate.py

By default this compares pred_affs, combined_affs and combined_affs_relabeled
(from 07_barcode_relabel.py), which should have the fewest splits. Pass
--affs to score others, e.g. pred_affs_raw or pseudo_affs if you segmented
those out of curiosity (not great, as you'd expect).

The labels were drawn inside this crop, so a neuron that leaves and comes
back can have two ids, and the barcode relabel joining those counts as a
merge. On this crop that is a handful of merges, so we just live with it.
--gt-merge merges labels with near identical barcodes before scoring if you
want to see the numbers without it (the merged groups are printed).

Keep in mind this is a quick in memory check on the crop the models were
trained on. The evaluation in the paper was done on larger regions with
skeleton ground truth and graph based metrics (expected run length, min cut
metric, ...), which is out of scope here.
"""

import csv

from funlib.persistence import open_ds

from prism_post.config import (
    COMBINED_AFFS,
    ENHANCED,
    LABELS,
    PRED_AFFS,
    argument_parser,
    crop_path,
    segments_name,
)
from prism_post.evaluate import (
    format_header,
    format_row,
    merge_gt_by_barcode,
    score,
)

DEFAULT_AFFS = [PRED_AFFS, COMBINED_AFFS, f"{COMBINED_AFFS}_relabeled"]


if __name__ == "__main__":
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--affs",
        nargs="+",
        default=DEFAULT_AFFS,
        help="affinity datasets whose segmentations to score (missing ones are skipped)",
    )
    parser.add_argument(
        "--csv", default=None, help="where to write results (default: <crop>/voi.csv)"
    )
    parser.add_argument(
        "--gt-merge",
        action="store_true",
        help="merge ground truth labels with near identical barcodes before scoring, see docstring",
    )
    args = parser.parse_args()

    sample = crop_path(args.data_dir, args.crop)
    csv_path = args.csv or sample / "voi.csv"
    labels = open_ds(sample / LABELS)
    gt = labels[labels.roi]
    if args.gt_merge:
        gt, groups = merge_gt_by_barcode(gt, open_ds(sample / ENHANCED)[labels.roi])
        print(
            f"merged {len(groups)} groups of labels with near identical barcodes: {groups}"
        )

    rows = []
    for affs in args.affs:
        store = sample / segments_name(affs)
        if not store.exists():
            print(f"{store} not found, skipping")
            continue
        seg = open_ds(store)[labels.roi]
        rows.append({"affs": affs, **score(gt, seg)})

    if not rows:
        raise SystemExit("no segmentations found, run 04_segment.py first")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + format_header())
    for r in rows:
        print(format_row(r["affs"], r))
    print(f"\nwrote {csv_path}")
