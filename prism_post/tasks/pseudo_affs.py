"""
Turn an embedding into "pseudo affinities".

For each affinity offset, compare a voxel's embedding with the embedding at
voxel + offset using cosine similarity. Two voxels from the same neuron have
the same barcode so they end up similar, neighboring neurons differ in a few
bits so they end up far apart. There is no boundary information in here at
all, which is exactly why it pairs well with predicted affinities (see
CombineAffs).
"""

from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from volara.blockwise import BlockwiseTask
from volara.datasets import Affs, Dataset, Raw
from volara.utils import PydanticCoordinate


def cosine_affinities(
    embedding: np.ndarray,
    neighborhood: list[Coordinate],
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    embedding is (c, z, y, x). Returns (len(neighborhood), z, y, x), where
    channel i is the cosine similarity between each voxel and the voxel at
    neighborhood[i] away. If that voxel is outside the array you get 0.
    """
    if normalize:
        norm = np.linalg.norm(embedding, axis=0, keepdims=True)
        embedding = embedding / np.maximum(norm, eps)

    out = np.zeros((len(neighborhood), *embedding.shape[1:]), dtype=np.float32)
    for i, offset in enumerate(neighborhood):
        # dot product of voxel v with voxel v + offset, where both exist
        base = [slice(None)] + [
            slice(max(-o, 0), -o if o > 0 else None) for o in offset
        ]
        shifted = [slice(None)] + [
            slice(max(o, 0), o if o < 0 else None) for o in offset
        ]
        sim = np.sum(embedding[tuple(base)] * embedding[tuple(shifted)], axis=0)
        out[(i, *base[1:])] = sim
    return out


class PseudoAffs(BlockwiseTask):
    """
    Cosine similarity of embedding_data along each offset in
    affs_data.neighborhood, written out as uint8 affinities.
    """

    task_type: Literal["pseudo_affs"] = "pseudo_affs"
    embedding_data: Raw
    """
    The embedding. Give it a scale_shift that maps the stored uint8 back to
    the real range, e.g. (2 / 255, -1) for [-1, 1].
    """
    affs_data: Affs
    """The output affinities. Its neighborhood gives the offsets to use."""
    block_size: PydanticCoordinate
    normalize: bool = True
    """Normalize embeddings to unit length before the dot product."""

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.affs_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.embedding_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.embedding_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> tuple[Coordinate, Coordinate]:
        # enough context on each side to reach every offset
        offsets = list(zip(*self.affs_data.neighborhood))
        low = Coordinate(max(-min(axis), 0) for axis in offsets) * self.voxel_size
        high = Coordinate(max(max(axis), 0) for axis in offsets) * self.voxel_size
        return low, high

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.affs_data]

    def drop_artifacts(self):
        self.affs_data.drop()

    def init(self):
        embedding = self.embedding_data.array("r")
        num_affs = len(self.affs_data.neighborhood)
        self.affs_data.prepare(
            shape=(num_affs, *(self.write_roi.shape / self.voxel_size)),
            chunk_shape=(num_affs, *self.block_size),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=embedding.units,
            axis_names=["affs^", *embedding.axis_names[1:]],
            types=["affs", *embedding.types[1:]],
            dtype=np.uint8,
        )

    @contextmanager
    def process_block_func(self):
        embedding = self.embedding_data.array("r")
        affs = self.affs_data.array("r+")

        def process_block(block):
            data = embedding.to_ndarray(block.read_roi, fill_value=0)
            sims = cosine_affinities(
                data, self.affs_data.neighborhood, normalize=self.normalize
            )
            sims = Array(sims, offset=block.read_roi.offset, voxel_size=self.voxel_size)
            write_roi = block.write_roi.intersect(affs.roi)
            affs[write_roi] = np.clip(sims[write_roi] * 255, 0, 255).astype(np.uint8)

        yield process_block
