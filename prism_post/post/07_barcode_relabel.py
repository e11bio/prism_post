"""
segments -> segments relabeled using barcodes

The segmentation from 04_segment.py is split leaning on purpose, so a neuron
that crosses a weak section or an affinity gap ends up as several segments.
Those pieces still carry the same barcode though, so we can stitch them back
together:

1. skeletonize the segments (kimimaro, teasar) and keep the skeleton endpoints
2. compute the mean barcode of every segment from the enhanced data
3. put the endpoints in a kd tree. Endpoints of two different segments
   within `match_radius` make those segments merge candidates. Endpoints
   within `split_factor * match_radius` make them split candidates
4. score each candidate pair with the cosine similarity of their barcodes
   minus `match_threshold`. Merge candidates become attractive or repulsive
   edges depending on the sign, split candidates only add repulsive edges
5. mutex watershed over those edges gives a segment -> segment mapping.
   Compose it with the fragment -> segment lut and relabel

    uv run python prism_post/post/07_barcode_relabel.py --affs combined_affs

Writes luts/<affs>_relabeled.npz and segments/<affs>_relabeled (or --name),
and prints VOI before and after. The skeletons are cached in skeletons/<affs>.npz since
that is the slow step, everything after it takes seconds. So it is cheap to
try a few thresholds and radii:

    uv run python prism_post/post/07_barcode_relabel.py --affs combined_affs --sweep

which scores every combination in memory and plots them (relabel_sweep.csv /
.png in the data dir) without writing a segmentation. --sweep-thresholds,
--sweep-radii and --sweep-split-factors set what gets tried, and --write also
writes every combination to segments/sweep/ so you can compare them in
neuroglancer with prism_post/view_sweep.py.
"""

import csv
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from funlib.persistence import open_ds
from scipy.spatial import KDTree
from volara.blockwise import Relabel
from volara.datasets import Labels
from volara.logging import set_log_basedir
from volara.lut import LUT

from prism_post.barcodes import (
    MATCH_RADIUS,
    MATCH_THRESHOLD,
    SPLIT_FACTOR,
    SWEEP_RADII,
    SWEEP_THRESHOLDS,
    Barcodes,
    apply_mapping,
    endpoints,
    load_skeletons,
    match_segments,
    save_skeletons,
    segment_barcodes,
    skeletonize,
)
from prism_post.blockwise import run
from prism_post.config import (
    BLOCK_SIZE,
    COMBINED_AFFS,
    ENHANCED,
    LABELS,
    SAMPLE,
    argument_parser,
    fragments_name,
    lut_name,
    segments_name,
    skeletons_name,
)
from prism_post.evaluate import format_header, format_row, score

# main


def plot_sweep(rows, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    settings = sorted(
        {(float(r["match_radius"]), float(r["split_factor"])) for r in rows}
    )
    for i, (radius, factor) in enumerate(settings):
        pts = sorted(
            (
                r
                for r in rows
                if float(r["match_radius"]) == radius
                and float(r["split_factor"]) == factor
            ),
            key=lambda r: float(r["match_threshold"]),
        )
        ax.plot(
            [float(r["voi_merge"]) for r in pts],
            [float(r["voi_split"]) for r in pts],
            marker="o",
            ms=4,
            lw=1.5,
            color=f"C{i}",
            label=f"{radius / 1000:g} um inner, {radius * factor / 1000:g} um outer",
        )
    before = rows[0]
    ax.plot(
        [float(before["voi_merge_before"])],
        [float(before["voi_split_before"])],
        "k*",
        ms=12,
        label="before relabel",
    )
    ax.set_xlabel("VOI merge")
    ax.set_ylabel("VOI split")
    ax.set_title(
        "points along a line are the match thresholds, lower left is better", fontsize=9
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argument_parser(__doc__, workers=8)
    parser.add_argument(
        "--affs", default=COMBINED_AFFS, help="which segmentation to relabel"
    )
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS, help="nm")
    parser.add_argument("--split-factor", type=float, default=SPLIT_FACTOR)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="score a grid of thresholds and radii, don't write a segmentation",
    )
    parser.add_argument(
        "--sweep-thresholds", nargs="+", type=float, default=SWEEP_THRESHOLDS
    )
    parser.add_argument(
        "--sweep-radii", nargs="+", type=float, default=SWEEP_RADII, help="nm"
    )
    parser.add_argument(
        "--sweep-split-factors", nargs="+", type=float, default=[SPLIT_FACTOR]
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="with --sweep: also write every combination to sweep/segments/ so you can look at them",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="name of the output segmentation (default: <affs>_relabeled)",
    )
    args = parser.parse_args()

    sample = args.data_dir / SAMPLE
    set_log_basedir(args.data_dir / "volara_logs")

    labels = open_ds(sample / LABELS)
    roi = labels.roi
    gt = labels[roi]
    segments_ds = open_ds(sample / segments_name(args.affs))
    segments = segments_ds[roi]
    voxel_size = segments_ds.voxel_size

    # 1. skeletons (cached, redone if the segmentation changed)
    segment_ids = np.unique(segments)
    skeleton_path = sample / skeletons_name(args.affs)
    skeletons = (
        load_skeletons(skeleton_path, segment_ids) if skeleton_path.exists() else None
    )
    if skeletons is not None:
        print(f"loaded skeletons from {skeleton_path}", flush=True)
    else:
        print(f"skeletonizing {len(segment_ids) - 1} segments ...", flush=True)
        skeletons = skeletonize(segments, voxel_size, roi.offset, args.num_workers)
        skeleton_path.parent.mkdir(exist_ok=True)
        save_skeletons(skeletons, skeleton_path, segment_ids)
        print(f"saved {len(skeletons)} skeletons to {skeleton_path}", flush=True)
    positions, end_labels = endpoints(skeletons)
    end_labels = end_labels.astype(np.int64)
    tree = KDTree(positions)
    print(f"{len(positions)} skeleton endpoints", flush=True)

    # 2. barcodes
    print("computing segment barcodes ...", flush=True)
    barcodes = Barcodes(*segment_barcodes(open_ds(sample / ENHANCED)[roi], segments))
    # segments too small to skeletonize have no endpoints, that is fine,
    # they just never become candidates

    frag_lut = LUT(path=sample / lut_name(args.affs)).load()

    def write_segmentation(mapping, out_name):
        """Compose the segment mapping with the fragment -> segment lut and relabel."""
        new_lut = np.stack([frag_lut[0], apply_mapping(frag_lut[1], mapping)])
        lut = LUT(path=sample / lut_name(out_name))
        lut.file.parent.mkdir(parents=True, exist_ok=True)
        lut.save(new_lut)
        run(
            Relabel(
                roi=roi,
                frags_data=Labels(store=sample / fragments_name(args.affs)),
                seg_data=Labels(store=sample / segments_name(out_name)),
                lut=lut,
                block_size=BLOCK_SIZE,
                num_workers=args.num_workers,
            )
        )
        print(f"wrote {sample / segments_name(out_name)}", flush=True)

    before = score(gt, segments)
    print(format_header(), flush=True)
    print(format_row("before", before), flush=True)

    if args.sweep:
        rows = []
        for radius, factor, threshold in product(
            args.sweep_radii, args.sweep_split_factors, args.sweep_thresholds
        ):
            mapping, stats = match_segments(
                barcodes, tree, end_labels, threshold, radius, factor
            )
            result = score(gt, apply_mapping(segments, mapping))
            rows.append(
                {
                    "affs": args.affs,
                    "match_threshold": threshold,
                    "match_radius": radius,
                    "split_factor": factor,
                    **result,
                    "voi_split_before": before["voi_split"],
                    "voi_merge_before": before["voi_merge"],
                    **stats,
                }
            )
            print(
                format_row(
                    f"r {radius / 1000:>4.1f}um x{factor:g} thr {threshold}", result
                )
                + f"   merged {stats['segments_merged']} segments into {stats['groups']}",
                flush=True,
            )
            if args.write:
                write_segmentation(
                    mapping,
                    f"sweep/{args.affs}_r{radius / 1000:g}um_x{factor:g}_t{threshold:g}",
                )
        csv_path = args.data_dir / "relabel_sweep.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path}", flush=True)
        plot_sweep(rows, args.data_dir / "relabel_sweep.png")
        raise SystemExit

    # 3. match + relabel
    mapping, stats = match_segments(
        barcodes,
        tree,
        end_labels,
        args.match_threshold,
        args.match_radius,
        args.split_factor,
    )
    print(stats, flush=True)
    print(format_row("after", score(gt, apply_mapping(segments, mapping))), flush=True)

    write_segmentation(mapping, args.name or f"{args.affs}_relabeled")
