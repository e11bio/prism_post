"""
The pieces of the barcode relabel: skeletons, segment barcodes, the spatial
queries and the mutex watershed over segments. Used by
post/07_barcode_relabel.py and post/06_sweep.py --relabel.
"""

import kimimaro
import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate
from scipy.spatial import KDTree

# cosine similarity above which two segments count as the same barcode. just
# used 0.5 here, but can sweep over them
MATCH_THRESHOLD = 0.5

# can sweep over, slightly smaller than in paper since on crop not larger roi
MATCH_RADIUS = 10000
SPLIT_FACTOR = 3

# skeletonize at full resolution. On this crop that only takes a minute or
# two. For the much larger regions in the paper we downsampled first, e.g.
# (1, 2, 2) makes the voxels roughly isotropic and is plenty for endpoints
# also unclear what optimal vals are for each of these params, kind of
# heuristically chosen
SKELETON_DOWNSAMPLE = Coordinate(1, 1, 1)
TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2,
    "max_paths": 20,
}

SWEEP_THRESHOLDS = [0.5, 0.7, 0.85, 0.95]
SWEEP_RADII = [2000, 5000, 10000]


# skeletons


def skeletonize(segments, voxel_size, offset, num_workers):
    """
    Skeletonize every segment. Returns {label: (vertices, edges)} with
    vertices in world coordinates (nm).
    """
    ds = tuple(slice(None, None, int(d)) for d in SKELETON_DOWNSAMPLE)
    skels = kimimaro.skeletonize(
        segments[ds],
        teasar_params=TEASAR_PARAMS,
        anisotropy=tuple(voxel_size * SKELETON_DOWNSAMPLE),
        dust_threshold=0,
        fix_branching=True,
        fix_borders=True,
        fill_holes=True,
        progress=True,
        parallel=num_workers,
        parallel_chunk_size=100,
    )
    return {
        int(label): (skel.vertices + np.array(offset), skel.edges)
        for label, skel in skels.items()
    }


def save_skeletons(skeletons, path, segment_ids):
    # segment_ids and the downsampling are stored so we can tell if the
    # skeletons are stale (e.g. 04_segment.py rerun with another bias)
    arrays = {"segment_ids": segment_ids, "downsample": np.array(SKELETON_DOWNSAMPLE)}
    for label, (vertices, edges) in skeletons.items():
        arrays[f"{label}_vertices"] = vertices
        arrays[f"{label}_edges"] = edges
    np.savez_compressed(path, **arrays)


def load_skeletons(path, segment_ids):
    """The cached skeletons, or None if they belong to a different segmentation."""
    data = np.load(path)
    if (
        "segment_ids" not in data.files
        or "downsample" not in data.files
        or not np.array_equal(data["segment_ids"], segment_ids)
        or not np.array_equal(data["downsample"], np.array(SKELETON_DOWNSAMPLE))
    ):
        return None
    labels = {
        int(k.rsplit("_", 1)[0])
        for k in data.files
        if k not in ("segment_ids", "downsample")
    }
    return {
        label: (data[f"{label}_vertices"], data[f"{label}_edges"]) for label in labels
    }


def endpoints(skeletons):
    """Skeleton nodes with at most one edge. Returns positions (n, 3) and labels (n,)."""
    positions, labels = [], []
    for label, (vertices, edges) in skeletons.items():
        degree = np.bincount(edges.ravel(), minlength=len(vertices))
        ends = vertices[degree <= 1]
        positions.append(ends)
        labels.append(np.full(len(ends), label))
    return np.concatenate(positions), np.concatenate(labels)


# barcodes


def segment_barcodes(barcodes, segments):
    """
    Mean barcode per segment. barcodes is (c, z, y, x), segments (z, y, x).
    Returns (ids, table) with table[i] the barcode of ids[i]: z-scored across
    segments so every channel counts the same, then mean centered and
    normalized so a dot product is a cosine similarity.
    """
    ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.float64)
    means = np.stack(
        [
            np.bincount(inverse, weights=channel.ravel(), minlength=len(ids)) / counts
            for channel in barcodes.astype(np.float32)
        ],
        axis=1,
    )
    keep = ids != 0
    ids, means = ids[keep], means[keep]
    means = (means - means.mean(axis=0)) / (means.std(axis=0) + 1e-8)
    means = means - means.mean(axis=1, keepdims=True)
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    return ids, means


class Barcodes:
    """Segment barcodes with a cosine similarity lookup by segment id."""

    def __init__(self, ids, table):
        self.ids = ids
        self.table = table
        self.row = {int(i): r for r, i in enumerate(ids)}

    def similarity(self, u, v):
        """Cosine similarity between the barcodes of two segments (arrays ok)."""
        return np.sum(self.table[u] * self.table[v], axis=-1)

    def rows(self, labels):
        return np.array([self.row[int(label)] for label in labels], dtype=np.int64)


# matching


def candidate_pairs(tree, labels, radius):
    """
    Pairs of different segments with endpoints within radius of each other,
    as an (n, 2) array of (smaller label, larger label), each pair once.
    """
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lu, lv = labels[pairs[:, 0]], labels[pairs[:, 1]]
    keep = lu != lv
    a, b = np.minimum(lu[keep], lv[keep]), np.maximum(lu[keep], lv[keep])
    # one int64 key per pair so np.unique is a cheap 1d sort
    key = np.unique(a.astype(np.int64) * (labels.max() + 1) + b)
    return np.stack([key // (labels.max() + 1), key % (labels.max() + 1)], axis=1)


def match_segments(barcodes, tree, labels, match_threshold, match_radius, split_factor):
    """
    Returns {old label: new label} for every segment that gets merged into
    another, plus a few numbers about what happened.
    """
    inner = candidate_pairs(tree, labels, match_radius)
    outer = candidate_pairs(tree, labels, match_radius * split_factor)
    width = labels.max() + 1
    outer_only = np.setdiff1d(
        outer[:, 0] * width + outer[:, 1], inner[:, 0] * width + inner[:, 1]
    )
    split = np.stack([outer_only // width, outer_only % width], axis=1)

    inner_w = (
        barcodes.similarity(barcodes.rows(inner[:, 0]), barcodes.rows(inner[:, 1]))
        - match_threshold
    )
    split_w = (
        barcodes.similarity(barcodes.rows(split[:, 0]), barcodes.rows(split[:, 1]))
        - match_threshold
    )
    split, split_w = split[split_w < 0], split_w[split_w < 0]

    pairs = np.concatenate([inner, split])
    weights = np.concatenate([inner_w, split_w])
    # strongest opinions first, that is the order mutex watershed wants
    order = np.argsort(-np.abs(weights))
    edges = [(bool(weights[i] > 0), int(pairs[i, 0]), int(pairs[i, 1])) for i in order]
    lut = mws.cluster(edges)

    # mws.cluster gives (node, root) pairs. Give every group that actually
    # merged something a fresh id so it can't collide with an existing one
    groups = {}
    for node, root in lut:
        groups.setdefault(root, []).append(node)
    next_id = int(barcodes.ids.max()) + 1
    mapping = {}
    for members in groups.values():
        if len(members) > 1:
            for node in members:
                mapping[node] = next_id
            next_id += 1

    stats = {
        "merge_candidates": len(inner),
        "split_candidates": len(split),
        "attractive_edges": int((inner_w > 0).sum()),
        "segments_merged": len(mapping),
        "groups": next_id - int(barcodes.ids.max()) - 1,
    }
    return mapping, stats


def apply_mapping(array, mapping):
    """Replace labels in array according to mapping, leaving the rest alone."""
    if not mapping:
        return array
    src = np.array(sorted(mapping), dtype=array.dtype)
    dst = np.array([mapping[s] for s in src], dtype=array.dtype)
    idx = np.searchsorted(src, array)
    idx[idx == len(src)] = 0
    hit = src[idx] == array
    out = array.copy()
    out[hit] = dst[idx[hit]]
    return out


def relabel_mapping(
    segments,
    barcodes_data,
    voxel_size,
    offset,
    num_workers=8,
    match_threshold=MATCH_THRESHOLD,
    match_radius=MATCH_RADIUS,
    split_factor=SPLIT_FACTOR,
    skeletons=None,
):
    """
    The whole thing in one call: {old segment id: new id} for the segments
    that get merged. Pass skeletons to skip skeletonizing.
    """
    if skeletons is None:
        skeletons = skeletonize(segments, voxel_size, offset, num_workers)
    positions, labels = endpoints(skeletons)
    barcodes = Barcodes(*segment_barcodes(barcodes_data, segments))
    mapping, stats = match_segments(
        barcodes,
        KDTree(positions),
        labels.astype(np.int64),
        match_threshold,
        match_radius,
        split_factor,
    )
    return mapping, stats, skeletons
