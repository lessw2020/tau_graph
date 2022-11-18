import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils._pytree import tree_flatten, tree_map

from functools import partial

from .graph_utils import (
    OP,
    get_node_tensor_numel,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
)
from .log_utils import rank0_debug

logger: logging.Logger = logging.getLogger(__name__)

_debug = partial(rank0_debug, logger)


# enum for the supported fusion comm types
class CommType(str, Enum):
    allreduce = "allreduce_"
    allgather = "allgather_"
    broadcast = "broadcast_"
    reducescatter = "reduce_scatter_"
    scatter = "scatter_"


@dataclass
class FusionElement:
    """This class tracks the nodes for a DTensor expanded communication collective in the graph"""

    in_graph: bool = False
    comm_type: Optional[CommType] = None
    node_list: Optional[List[fx.Node]] = field(default_factory=lambda: [])  # type: ignore
    size: int = 0
    prev_node: Optional[fx.Node] = None  # node that was before start of section
    next_node: Optional[fx.Node] = None  # node that was after end of section
    processed: bool = False
    output_name: str = ""
    clone_node: Optional[fx.Node] = None
    comm_node: Optional[fx.Node] = None
    wait_node: Optional[fx.Node] = None


@dataclass
class GraphInfo:
    len: int = 0
    global_buffer_node: Optional[fx.Node] = None
    global_buffer_size: int = 0
    output: Optional[fx.Node] = None
    first: Optional[fx.Node] = None
    placeholder_nodes: Optional[fx.Node] = field(default_factory=lambda: [])

    def update_info(self, gm: fx.GraphModule) -> None:
        """get the len, input and output nodes"""
        graph_len = gm.graph._len
        if not graph_len:
            raise ValueError("Empty graph passed in....")
        self.len = graph_len

        nodelist = gm.graph.nodes

        for i, node in enumerate(nodelist):
            if node.op == OP.PLACEHOLDER:
                self.placeholder_nodes.append(node)

        self.first_placeholder = self.placeholder_nodes[0]

        self.output = get_output_node(gm)
        assert (
            self.output is not None
        ), f"unable to locate output node in gm {gm.graph}"

        rank0_debug(
            logger,
            f"global graph_info - len = {self.len} input = {self.first_placeholder}, output = {self.output}",
        )


def _insert_fusion_buffer_node(
    gm: fx.GraphModule, gi: GraphInfo, buffer_size: Iterable[int]
) -> fx.Node:
    """insert a torch.empty node after last placeholder node"""
    insert_after_node = gi.placeholder_nodes[-1]
    rank0_debug(
        logger,
        f"\n----> global buffer insert after node = {insert_after_node.name}",
    )

    with gm.graph.inserting_after(insert_after_node):
        new_buffer_node = gm.graph.create_node(
            OP.CALL_FUNCTION,
            target=torch.empty,
            # TODO - need device from DTensor to put buffer on gpu
            args=(buffer_size,),
        )
    assert (
        new_buffer_node is not None
    ), f"failed to create buffer node, size={buffer_size}"

    gi.global_buffer_node = new_buffer_node

    return new_buffer_node


def _insert_fusion_loader(
    gm: fx.GraphModule,
    gi: GraphInfo,
    list_to_fuse: list[FusionElement],
    fuse_location: int,
    buffer_size: int,
):
    """given a list of fusion candidates, insert the buffer load portion into the graph
    fuse_location = index in the list to fuse for where to insert the loading"""
    pass


def _scan_graph_for_fusion_elements(
    gm: fx.GraphModule,
    comm_type: CommType = CommType.allreduce,
) -> Optional[List[FusionElement]]:
    """scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match CommType"""

    element_list = []

    fe_sequence = [
        "clone",
        "_tensor_constant",
        "_tensor_constant",
        comm_type,
        "comm_result",
        "getitem",
        "getitem",
        "wait_comm",
    ]

    fe_size = len(fe_sequence) - 1
    index = 0
    curr_node_list = []

    for i, node in enumerate(gm.graph.nodes):
        pattern = fe_sequence[index]

        if index < fe_size:
            if node.name.startswith(pattern):
                curr_node_list.append(node)
                index += 1
                continue
            else:
                index = 0
                curr_node_list.clear()
                continue

        elif index == fe_size:
            # should be last node
            if node.name.startswith(pattern):
                curr_node_list.append(node)

                fe = FusionElement(
                    comm_type=comm_type, node_list=deepcopy(curr_node_list)
                )

                # need to fully populate this fe...
                # we will be removing/rewriting the node list so we save prev and next
                fe.prev_node = curr_node_list[0].prev
                fe.next_node = node.next

                fe.output_name = node.name
                fe.wait_node = node

                fe.clone_node = curr_node_list[0]
                fe.comm_node = curr_node_list[3]

                fe.size = get_node_tensor_numel(fe.clone_node)  # type: ignore
                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def _fuse_elements(
    left: FusionElement,
    right: FusionElement,
    gm: fx.GraphModule,
) -> bool:
    """takes two fusion elements and merges them into a single graph comm operation"""

    def get_shape(node):
        tdata = node.meta.get("tensor_meta")
        m, n = tdata.shape
        return (m, n)

    rank0_debug(logger, f"fe inspection {left=}")
    rank0_debug(logger, f"right inspection {right=}")

    left_shape = get_shape(left.clone_node)
    left_size = left.size
    right_shape = get_shape(right.clone_node)
    right_size = right.size

    rank0_debug(
        logger, f"left shape = {left_shape}, right_shape = {right_shape}"
    )

    buffer_size = left.size + right.size

    def load_buffer(
        buffer,
        buffer_size,
        left,
        left_size,
        right,
        right_size,
    ):

        buffer[0:left_size] = left.view(-1)
        buffer[left_size:buffer_size] = right.view(-1)

    buffer = torch.empty(buffer_size)

    left_tensor = torch.randn(left_shape)
    right_tensor = torch.randn(right_shape)

    traced = make_fx(load_buffer)(
        buffer,
        buffer_size,
        left_tensor,
        left_size,
        right_tensor,
        right_size,
    )

    rank0_debug(logger, f"traced = {traced.graph}")

    def unpack_buffer(
        buffer,
        left,
        left_size,
        left_shape,
        right,
        right_size,
        right_shape,
    ):
        left.copy_(buffer[0:left_size].view(left_shape))

        right.copy_(
            buffer[left_size : left_size + right_size].view(right_shape)
        )

    traced_unpack = make_fx(unpack_buffer)(
        buffer,
        left_tensor,
        left_size,
        left_shape,
        right_tensor,
        right_size,
        right_shape,
    )

    rank0_debug(logger, f"traced = {traced.graph}")
    rank0_debug(logger, f" unpack graph\n {traced_unpack.graph}")

    return True


def _create_node_map(nodelist):
    newdict = {}
    for node in nodelist:
        newdict[node.name] = node
    return newdict


def _map_nodes(gm):
    mapping = {}
    for node in gm.graph.nodes:
        mapping[node.name] = node
    return mapping


def _remove_gradient_tensor_clones(gm: fx.GraphModule) -> int:
    """Optimizes away any duplicate gradient tensor nodes from the provided graph.
    Returns - total count of clone tensor nodes removed"""

    count_clones_removed = 0
    sequence = ["clone", "_tensor_constant0", "_tensor_constant1", "allreduce_"]
    len_sequence = len(sequence) - 1
    _debug(f"{len_sequence=}")
    index = 0
    clone_node = None
    comm_node = None

    for i, node in enumerate(gm.graph.nodes):

        if node.op == OP.PLACEHOLDER:
            index = 0
            continue

        pattern = sequence[index]
        _debug(f"in loop -> index = {index}, node = {node.name}, {pattern=}")

        if (
            index == 0
            and node.op == OP.CALL_FUNCTION
            and node.name.startswith(pattern)
        ):
            clone_node = node
            index += 1
            continue

        elif index < len_sequence and node.name.startswith(pattern):
            index += 1
            continue

        elif (
            index == len_sequence
            and node.op == OP.CALL_FUNCTION
            and node.name.startswith(pattern)
        ):
            # found matching clone/comm sequence...
            grad_tensor_node = clone_node.args[0]
            comm_node = node
            _debug(
                f"preparing to relink - {grad_tensor_node.name} to {comm_node.name}, removing clone {clone_node.name}, {clone_node.args}"
            )
            comm_node.update_arg(0, [grad_tensor_node])

            # reset for next series
            count_clones_removed += 1
            index = 0
        else:
            # failed sequence
            index = 0

    if count_clones_removed:
        graph_cleanup(gm)
    _debug(
        f"_clean_grad_tensor_clones removed {count_clones_removed} clone nodes..."
    )
    return count_clones_removed

    nodemap = _map_nodes(gm)  # create_node_map(curr_fe.node_list)

    debug(f"node map via partial = {nodemap}")

    #
    debug(f"current graph = {gm.graph.print_tabular()}\n")
    allreduce_node = nodemap["allreduce__default"]
    debug(f"allreduce node = {allreduce_node.name}")
    t14node = nodemap["t_14"]  # curr_fe.prev_node
    debug(f"allreduce args and type {[type(x) for x in allreduce_node.args]}")
    allreduce_node.update_arg(0, [t14node])

    debug(f"all_reduce new args = {allreduce_node.args}\n")

    # second one
    allreduce_node1 = nodemap["allreduce__default_1"]
    debug(f"allreduce node = {allreduce_node1.name}")
    t11node = nodemap["t_11"]  # curr_fe.prev_node
    debug(f"allreduce args and type {[type(x) for x in allreduce_node1.args]}")
    allreduce_node1.update_arg(0, [t11node])

    gm.recompile()
    debug(f"{gm.graph}\n")


def run_comm_fusion(gm: fx.GraphModule) -> Optional[fx.GraphModule]:
    """main entry into remapping graph for all_reduce fusion"""

    rank0_debug(logger, "entered main comm_fusion run 134...\n")
    result = False

    # get our main graph info
    graph_info = GraphInfo()
    graph_info.update_info(gm)
    rank0_debug(logger, f"graph info {graph_info}")

    _debug(f"graph pre clone node optimization {gm.graph.print_tabular()}")

    clones_removed = _remove_gradient_tensor_clones(gm)

    _debug(
        f"\n\n\n===> graph pre clone node optimization {gm.graph.print_tabular()}"
    )

    # scan graph for all comm sections (fusion elements)
    # fe_list = _scan_graph_for_fusion_elements(gm, comm_type=CommType.allreduce)

    # TODO - determine optimal reusable buffer size...for this test, use 200

    # global_fusion_buffer = _insert_fusion_buffer_node(gm, graph_info, 200)

    # rank0_debug(logger, f"added global fusion_buffer node")
    # rank0_debug(logger, f"\n{gm.graph}\n")

    # clean up subgraph
    # _clean_wait_graph(gm, fe_list)

    # start fusion
    # res = _fuse_elements(fe_list[0], fe_list[1], gm)

    # final review print
    graph_cleanup(gm)

    pretty_print_graph(gm, "final version, fusion pass")

    result = True  # TODO - make this mean something
    return gm
