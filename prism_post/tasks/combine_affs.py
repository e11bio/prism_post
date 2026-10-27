"""
Multiply two sets of affinities together.

We use it to combine predicted affinities (good at boundaries, but can merge
neighbors with similar barcodes) with pseudo affinities (good at barcodes, no
idea about boundaries). An edge has to look good to both to survive.
"""

from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import model_validator
from volara.blockwise import BlockwiseTask
from volara.datasets import Affs, Dataset
from volara.utils import PydanticCoordinate


class CombineAffs(BlockwiseTask):
    """out_affs = affs_a * affs_b, all uint8."""

    task_type: Literal["combine_affs"] = "combine_affs"
    affs_a: Affs
    affs_b: Affs
    out_affs: Affs
    block_size: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @model_validator(mode="after")
    def _same_neighborhood(self) -> "CombineAffs":
        if not (
            self.affs_a.neighborhood
            == self.affs_b.neighborhood
            == self.out_affs.neighborhood
        ):
            raise ValueError("affs_a, affs_b and out_affs must share one neighborhood")
        return self

    @property
    def task_name(self) -> str:
        return f"{self.out_affs.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.affs_a.array("r").roi.intersect(self.affs_b.array("r").roi)
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.affs_a.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.voxel_size.dims)

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.out_affs]

    def drop_artifacts(self):
        self.out_affs.drop()

    def init(self):
        affs_a = self.affs_a.array("r")
        num_affs = len(self.out_affs.neighborhood)
        self.out_affs.prepare(
            shape=(num_affs, *(self.write_roi.shape / self.voxel_size)),
            chunk_shape=(num_affs, *self.block_size),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=affs_a.units,
            axis_names=affs_a.axis_names,
            types=affs_a.types,
            dtype=np.uint8,
        )

    @contextmanager
    def process_block_func(self):
        affs_a = self.affs_a.array("r")
        affs_b = self.affs_b.array("r")
        out = self.out_affs.array("r+")

        def process_block(block):
            roi = block.write_roi.intersect(out.roi)
            a = affs_a.to_ndarray(roi).astype(np.float32) / 255
            b = affs_b.to_ndarray(roi).astype(np.float32) / 255
            out[roi] = np.clip(a * b * 255, 0, 255).astype(np.uint8)

        yield process_block
