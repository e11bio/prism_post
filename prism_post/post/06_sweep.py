"""
Sweep the global mutex watershed bias and plot VOI.

The last two steps of the segmentation (graph mutex watershed + relabel) are
fast, so it is cheap to try a few biases and see where you land. This reuses
the fragments and graph from 04_segment.py, so run that first for each set
of affinities you want to compare.

The plot has VOI merge on x and VOI split on y, one line per affinities with
a point per bias. Closer to the bottom left is better.

    uv run python prism_post/post/06_sweep.py
    uv run python prism_post/post/06_sweep.py --biases -0.4,-0.7 -0.2,-0.7
    uv run python prism_post/post/06_sweep.py --relabel     # also run the barcode relabel on top of each bias
    uv run python prism_post/post/06_sweep.py --plot-only   # just redraw from the csv

--relabel adds a line per affinities with the barcode relabel (07) applied to
each bias, so you can see how far it moves things and how it interacts with
the bias. It skeletonizes every segmentation, so it takes a minute per bias.

Segmentations go under segments/sweep/, results to sweep.csv and sweep.png
in the crop zarr. If interrupted it picks up from the csv.
"""

import csv

import matplotlib.pyplot as plt
from funlib.persistence import open_ds
from volara.blockwise import GraphMWS, Relabel
from volara.datasets import Labels
from volara.dbs import SQLite
from volara.logging import set_log_basedir
from volara.lut import LUT

from prism_post.barcodes import apply_mapping, relabel_mapping
from prism_post.blockwise import run
from prism_post.config import (
    BLOCK_SIZE,
    COMBINED_AFFS,
    ENHANCED,
    LABELS,
    PRED_AFFS,
    argument_parser,
    crop_path,
    db_name,
    fragments_name,
    lut_name,
    segments_name,
)
from prism_post.evaluate import FIELDS as SCORE_FIELDS
from prism_post.evaluate import score

DEFAULT_AFFS = [PRED_AFFS, COMBINED_AFFS]

# (short range, long range) biases, from split leaning to merge leaning
BIASES = [(-0.6, -0.9), (-0.4, -0.7), (-0.2, -0.7), (-0.1, -0.5), (-0.05, -0.4)]

FIELDS = ["affs", "adj_bias", "lr_bias", *SCORE_FIELDS]


def tag(bias):
    return f"a{bias[0]}_l{bias[1]}"


def parse_bias(s):
    adj, lr = s.split(",")
    return float(adj), float(lr)


def read_results(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def append_result(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def plot(rows, path, biases, affs_names):
    """
    VOI merge against VOI split, one line per affinities with a point per
    bias. If the sweep was run with --relabel there is a second panel with
    the barcode relabel applied, on the same axes so you can compare.
    """
    order = {tag(b): i for i, b in enumerate(biases)}
    base_names = [a for a in affs_names if not a.endswith(" + relabel")]
    with_relabel = any(a.endswith(" + relabel") for a in affs_names) and any(
        r["affs"].endswith(" + relabel") for r in rows
    )
    panels = [("no barcode relabel", "")] + (
        [("with barcode relabel", " + relabel")] if with_relabel else []
    )
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6 * len(panels), 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (title, suffix) in zip(axes[0], panels):
        for i, affs in enumerate(base_names):
            pts = [r for r in rows if r["affs"] == affs + suffix]
            pts.sort(
                key=lambda r: order.get(
                    tag((float(r["adj_bias"]), float(r["lr_bias"]))), 99
                )
            )
            if not pts:
                continue
            ax.plot(
                [float(r["voi_merge"]) for r in pts],
                [float(r["voi_split"]) for r in pts],
                marker="o",
                ms=4,
                lw=1.5,
                color=f"C{i}",
                label=affs,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("VOI merge")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("VOI split")
    fig.suptitle(
        "lower left is better, points along a line are the biases", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argument_parser(__doc__, workers=8)
    parser.add_argument(
        "--affs", nargs="+", default=DEFAULT_AFFS, help="affinities to compare"
    )
    parser.add_argument(
        "--biases",
        nargs="+",
        type=parse_bias,
        default=BIASES,
        metavar="ADJ,LR",
        help="(short range, long range) biases, e.g. -0.2,-0.7",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="also score each bias with the barcode relabel on top",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    sample = crop_path(args.data_dir, args.crop)
    csv_path = sample / "sweep.csv"
    png_path = sample / "sweep.png"
    set_log_basedir(sample / "volara_logs")

    series = (
        args.affs + [f"{a} + relabel" for a in args.affs] if args.relabel else args.affs
    )
    if args.plot_only:
        plot(read_results(csv_path), png_path, args.biases, series)
        raise SystemExit

    labels = open_ds(sample / LABELS)
    roi = labels.roi
    gt = labels[roi]
    enhanced = open_ds(sample / ENHANCED)[roi] if args.relabel else None
    done = {
        (r["affs"], tag((float(r["adj_bias"]), float(r["lr_bias"]))))
        for r in read_results(csv_path)
    }

    for affs in args.affs:
        if not (sample / fragments_name(affs)).exists():
            print(
                f"no fragments for {affs}, run 04_segment.py --affs {affs} first, skipping"
            )
            continue
        fragments = Labels(store=sample / fragments_name(affs))
        db = SQLite(
            path=sample / db_name(affs),
            edge_attrs={"adj_weight": "float", "lr_weight": "float"},
        )
        for bias in args.biases:
            if (affs, tag(bias)) in done and (
                not args.relabel or (f"{affs} + relabel", tag(bias)) in done
            ):
                continue
            print(f"[{affs}] bias {bias}", flush=True)
            name = f"sweep/{affs}_{tag(bias)}"
            lut = LUT(path=sample / lut_name(name))
            lut.file.parent.mkdir(parents=True, exist_ok=True)
            segments = Labels(store=sample / segments_name(name))
            run(
                GraphMWS(
                    roi=roi,
                    db=db,
                    lut=lut,
                    weights={"adj_weight": (1.0, bias[0]), "lr_weight": (1.0, bias[1])},
                ),
                multiprocessing=False,
            )
            run(
                Relabel(
                    roi=roi,
                    frags_data=fragments,
                    seg_data=segments,
                    lut=lut,
                    block_size=BLOCK_SIZE,
                    num_workers=args.num_workers,
                )
            )
            seg = segments.array()[roi]
            if (affs, tag(bias)) not in done:
                result = score(gt, seg)
                append_result(
                    csv_path,
                    {"affs": affs, "adj_bias": bias[0], "lr_bias": bias[1], **result},
                )
                print(f"  {result}", flush=True)
            if args.relabel:
                mapping, stats, _ = relabel_mapping(
                    seg, enhanced, labels.voxel_size, roi.offset, args.num_workers
                )
                result = score(gt, apply_mapping(seg, mapping))
                append_result(
                    csv_path,
                    {
                        "affs": f"{affs} + relabel",
                        "adj_bias": bias[0],
                        "lr_bias": bias[1],
                        **result,
                    },
                )
                print(f"  + relabel: {result}", flush=True)
            plot(read_results(csv_path), png_path, args.biases, series)
