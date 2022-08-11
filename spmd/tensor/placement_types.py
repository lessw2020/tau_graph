# Copyright (c) Meta Platforms, Inc. and affiliates
import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass
from typing import Optional, List, Sequence, cast
from spmd.tensor.device_mesh import DeviceMesh


@dataclass
class Placement(object):
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


@dataclass
class Shard(Placement):
    # shard placement, shard on a dim
    dim: int


@dataclass
class Replicate(Placement):
    # replicate placement
    pass


@dataclass
class _Partial(Placement):
    # partial placement with reduce op
    reduce_op: c10d.ReduceOp = c10d.ReduceOp.SUM


# used internally to propagate the placements
@dataclass
class PlacementSpec(object):
    # number of dimensions (rank) of the current dist tensor
    ndim: int
    mesh: DeviceMesh
    placements: Sequence[Placement]

    @property
    def dim_map(self) -> List[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 1, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a PlacementSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                r[shard_dim] = i
        return r

    @classmethod
    def from_dim_map(
        cls, mesh: DeviceMesh, dim_map: List[int], sums: List[int]
    ) -> "PlacementSpec":
        """
        Construct a PlacementSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the PlacementSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.

        Return:
            a class:`PlacementSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        for i, m in enumerate(dim_map):
            if m >= 0:
                if not placements[m].is_replicate():
                    raise RuntimeError(
                        "DeviceMesh cann't be mapped to two dimension of the same tensor"
                    )
                placements[m] = Shard(i)

        for s in sums:
            placements[s] = _Partial()

        spec = cls(len(dim_map), mesh, placements)
        return spec