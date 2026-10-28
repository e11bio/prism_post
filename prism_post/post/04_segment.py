"""
affinities -> segmentation

The usual volara pipeline, run inside the labels roi so we can score it:

1. ExtractFrags - mutex watershed in each block gives us fragments (supervoxels)
   and a node per fragment in a sqlite graph. Affinities outside the mask are ignored
2. AffAgglom - for each pair of touching fragments, score the edge with the mean
   short range affinity (adj_weight) and mean long range affinity (lr_weight)
3. GraphMWS - global mutex watershed on that graph, gives a fragment -> segment lut
4. Relabel - apply the lut to the fragments

    uv run python prism_post/post/04_segment.py --affs combined_affs

Run it once per set of affinities you want to compare (pred_affs,
pred_affs_raw, pseudo_affs, combined_affs). Outputs are named after the
affinities: fragments/<affs>, segments/<affs>, dbs/<affs>.sqlite, luts/<affs>.npz
"""

from funlib.persistence import open_ds
from volara.blockwise import AffAgglom, ExtractFrags, GraphMWS, Relabel
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite
from volara.logging import set_log_basedir
from volara.lut import LUT

from prism_post.blockwise import run
from prism_post.config import (
    BLOCK_SIZE,
    COMBINED_AFFS,
    CONTEXT,
    FRAG_BIAS,
    LABELS,
    MASK,
    N_ADJ,
    NEIGHBORHOOD,
    RANDOMIZED_STRIDES,
    REMOVE_DEBRIS,
    SEG_BIAS,
    STRIDES,
    affs_name,
    argument_parser,
    crop_path,
    db_name,
    fragments_name,
    lut_name,
    segments_name,
)

if __name__ == "__main__":
    parser = argument_parser(__doc__, workers=8)
    parser.add_argument(
        "--affs",
        default=COMBINED_AFFS,
        help=f"affinity dataset to segment (default: {COMBINED_AFFS})",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="do not mask affinities with the binary mask",
    )
    args = parser.parse_args()

    sample = crop_path(args.data_dir, args.crop)
    set_log_basedir(sample / "volara_logs")

    # only segment where we have ground truth to score against
    roi = open_ds(sample / LABELS).roi

    affs = Affs(store=sample / affs_name(args.affs), neighborhood=NEIGHBORHOOD)
    mask = None if args.no_mask else Raw(store=sample / MASK)
    fragments = Labels(store=sample / fragments_name(args.affs))
    segments = Labels(store=sample / segments_name(args.affs))
    db = SQLite(
        path=sample / db_name(args.affs),
        edge_attrs={"adj_weight": "float", "lr_weight": "float"},
    )
    lut = LUT(path=sample / lut_name(args.affs))
    lut.file.parent.mkdir(parents=True, exist_ok=True)

    extract_frags = ExtractFrags(
        roi=roi,
        db=db,
        affs_data=affs,
        frags_data=fragments,
        mask_data=mask,
        block_size=BLOCK_SIZE,
        context=CONTEXT,
        bias=FRAG_BIAS,
        strides=STRIDES,
        randomized_strides=RANDOMIZED_STRIDES,
        remove_debris=REMOVE_DEBRIS,
        num_workers=args.num_workers,
    )

    aff_agglom = AffAgglom(
        roi=roi,
        db=db,
        affs_data=affs,
        frags_data=fragments,
        block_size=BLOCK_SIZE,
        context=CONTEXT,
        scores={
            "adj_weight": NEIGHBORHOOD[:N_ADJ],
            "lr_weight": NEIGHBORHOOD[N_ADJ:],
        },
        num_workers=args.num_workers,
    )

    graph_mws = GraphMWS(
        roi=roi,
        db=db,
        lut=lut,
        weights={name: (1.0, bias) for name, bias in SEG_BIAS.items()},
    )

    relabel = Relabel(
        roi=roi,
        frags_data=fragments,
        seg_data=segments,
        lut=lut,
        block_size=BLOCK_SIZE,
        num_workers=args.num_workers,
    )

    run(extract_frags)
    run(aff_agglom)
    run(graph_mws, multiprocessing=False)  # one global step, no workers needed
    run(relabel)
