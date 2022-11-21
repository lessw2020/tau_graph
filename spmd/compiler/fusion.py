import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional
from functools import partial
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot

import torch
import torch.fx as fx

from .graph_utils import (
    OP,
    get_node_tensor_numel_shape,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
    create_graph_node_map,
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
    shape: Optional[List[int]] = field(default_factory=lambda: [])  # type: ignore
    prev_node: Optional[fx.Node] = None  # node that was before start of section
    next_node: Optional[fx.Node] = None  # node that was after end of section
    processed: bool = False
    output_name: str = ""
    comm_node: Optional[fx.Node] = None
    wait_node: Optional[fx.Node] = None
    grad_tensor_node: Optional[fx.Node] = None


@dataclass
class GraphInfo:
    len: int = 0
    global_buffer: Optional[fx.Node] = None
    global_buffer_size: int = 0
    output: Optional[fx.Node] = None
    first: Optional[fx.Node] = None

    def update_info(self, gm: fx.GraphModule) -> None:
        """get the len, input and output nodes"""
        graph_len = gm.graph._len
        if not graph_len:
            raise ValueError("Empty graph passed in....")
        self.len = graph_len

        nodelist = gm.graph.nodes

        for i, node in enumerate(nodelist):
            if node.op == OP.PLACEHOLDER:
                self.first = node
                break

        self.output = get_output_node(gm)
        assert (
            self.output is not None
        ), f"unable to locate output node in gm {gm.graph}"

        rank0_debug(
            logger,
            f"updated graph_info - len = {self.len} input = {self.first}, output = {self.output}",
        )


def _insert_fusion_buffer_node(
    gm: fx.GraphModule, buffer_size: Iterable[int]
) -> fx.Node:
    """insert a torch.empty node in front of insert_before_node"""

    # default to inserting just after last placeholder node
    for node in gm.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            continue
        insert_before_node = node
        _debug(f"\n{insert_before_node.name=}\n")
        break

    with gm.graph.inserting_before(insert_before_node):
        new_buffer_node = gm.graph.create_node(
            OP.CALL_FUNCTION,
            target=torch.empty,
            # TODO - need device from DTensor to put buffer on gpu
            args=(buffer_size,),
        )
    assert (
        new_buffer_node is not None
    ), f"failed to create buffer node, size={buffer_size}"
    _debug(f"{new_buffer_node=}\n")

    return new_buffer_node


def _scan_graph_for_fusion_elements(
    gm: fx.GraphModule,
    comm_type: CommType = CommType.allreduce,
) -> Optional[List[FusionElement]]:
    """scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match CommType"""

    element_list = []

    fe_sequence = [
        # "clone",
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
                _debug(f"prev node = {fe.prev_node}")
                fe.next_node = node.next

                fe.output_name = node.name
                fe.wait_node = node
                fe.comm_node = curr_node_list[2]

                fe.grad_tensor_node = fe.comm_node.args[0][0]

                size, shape = get_node_tensor_numel_shape(fe.grad_tensor_node)  # type: ignore
                fe.size = size
                fe.shape = shape
                _debug(f"\nfe list size shape {size=}, {shape=}\n")

                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def _copy_fe_to_buffer(
    gi: GraphInfo, gm: fx.GraphModule, fe_list: list[FusionElement]
) -> None:
    """first half of fusion - move desired items to buffer and create graph"""
    buffer_node = gi.global_buffer
    buffer_size = gi.global_buffer_size

    copy_list = fe_list

    def copy_to_buffer(buffer, tensor_list):
        offset = 0
        for t in tensor_list:
            size = t.numel()
            buffer[offset : offset + size] = t.view(-1)
            offset += size
        return buffer

    # setup dummy vars
    buffer = torch.empty(buffer_size)
    tlist = []
    for item in copy_list:
        a = torch.zeros((item.shape[0], item.shape[1]))
        tlist.append(a)
    _debug(f"\n++++ tlist ++++ \n{len(tlist)}")

    load_gm = make_fx(copy_to_buffer)(buffer, tlist)

    _debug(f"^^^^^^\n {load_gm.graph.print_tabular()}\n")

    subnodemap = create_graph_node_map(load_gm)

    # update load loop to use main graph items
    fn_list = []
    pl_list = []
    for node in load_gm.graph.nodes:
        _debug(f"f221 node {node.name}")
        if node.op == OP.PLACEHOLDER:
            pl_list.append(node)
        elif node.op == OP.CALL_FUNCTION:
            fn_list.append(node)

    # create placeholder remapping
    pl_map = {}
    pl_map[pl_list[0]] = gi.global_buffer
    pl_map[pl_list[1]] = fe_list[0].grad_tensor_node
    pl_map[pl_list[2]] = fe_list[1].grad_tensor_node

    insert_node = fe_list[-1].prev_node
    _debug(f" f229 insert after node = {insert_node.name}\n")

    # verify first node
    for node in gm.graph.nodes:
        if node.name == insert_node.name:
            true_insert_node = node

    _debug(
        f" compare insert nodes for {insert_node.name}, true {id(true_insert_node)} vs fe list {id(insert_node)}"
    )

    def remap_args(node: fx.Node):
        _debug(
            f"f242 -  callback with  arg {node=}, type {type(node)} and args {node.args=} "
        )
        if node in pl_map:
            original = node
            node = pl_map[node]
            _debug(f"f251 remapping {original} to {node} ")
        return node

    value_remap = {}
    with gm.graph.inserting_after(true_insert_node):
        for innernode in reversed(load_gm.graph.nodes):
            if innernode.op in [OP.PLACEHOLDER, OP.OUTPUT]:
                continue
            print(f"about to insert {innernode.name}")
            value_remap[innernode] = gm.graph.node_copy(innernode, remap_args)

    # insert into main graph, just above last fe
    _debug(f"f260 = {value_remap=}\n")
    _debug(f"f261remapping1  ^^$$\n {gm.graph.print_tabular()}")


def _scatter_results_from_buffer(gi, gm, fe_list):
    """after comm event with buffer, scatter results back to original fe tensors"""

    buffer_node = gi.global_buffer
    buffer_size = gi.global_buffer_size

    scatter_list = fe_list

    def scatter_from_buffer(buffer, scatter_list):
        offset = 0
        for t in scatter_list:
            numel = t.numel()
            shaper = buffer[offset : offset + numel].view(t.shape)
            t.copy_(shaper)  # buffer[offset : offset + numel].view(t.shape() ))
            offset += numel
        return buffer

    buffer = torch.empty(buffer_size)
    # _debug(f"buffer shape {buffer.shape}")
    tlist = []
    for item in scatter_list:
        shape = item.shape

        a = torch.zeros(item.shape[0], item.shape[1])

        tlist.append(a)  # clone().detach())

    scatter_sg = make_fx(scatter_from_buffer)(buffer, tlist)
    _debug(f"==== {scatter_sg.graph.print_tabular()}\n")

    #
    insert_node = fe_list[-1].wait_node
    _debug(f" f88 copy to insert node = {insert_node.name}\n")
    value_remap = {}

    # verify first node
    for node in gm.graph.nodes:
        if node.name == insert_node.name:
            true_insert_node = node

    _debug(
        f" compare insert nodes, true {id(true_insert_node)} vs fe list {id(insert_node)}"
    )

    with gm.graph.inserting_after(true_insert_node):
        for innernode in scatter_sg.graph.nodes:
            if innernode.op in [OP.PLACEHOLDER, OP.OUTPUT]:
                continue
            print(f"about to insert {innernode.name}")
            value_remap[innernode] = gm.graph.node_copy(
                innernode  # , lambda x: value_remap[x]
            )

    # insert into main graph, just above last fe
    _debug(f"f310  ^^$$\n {gm.graph.print_tabular()}")


def run_comm_fusion(gm: fx.GraphModule) -> bool:
    """main entry into remapping graph for all_reduce fusion"""

    result = False

    # first recompile to make sure we have coherent graph
    gm.recompile()

    # get our main graph info
    gi = GraphInfo()
    gi.update_info(gm)

    _debug(f"{gm.graph.print_tabular()}")

    # _debug(f"\n----- fe_list {len(fe_list)} -------- \n {fe_list}\n")

    # compute optimal buffer size here.... # TODO
    test_buffer_size = 200

    buffer_node = _insert_fusion_buffer_node(gm, test_buffer_size)

    # scan graph for all comm sections (fusion elements)
    fe_list = _scan_graph_for_fusion_elements(gm, comm_type=CommType.allreduce)

    gi.global_buffer = buffer_node
    _debug(
        f"f345 inserted buffer node {gi.global_buffer.name}, {gi.global_buffer.target}"
    )
    gi.global_buffer_size = test_buffer_size

    # copy fe_items to buffer
    _copy_fe_to_buffer(gi, gm, fe_list[:2])

    # TODO: use an all_reduce
    second_ar = fe_list[1].comm_node
    _debug(f"second ar tensor {second_ar.args[0][0].name}\n")

    second_ar.update_arg(0, [buffer_node])
    _debug(f"{second_ar.args=}")
    _debug(
        f"\n$$$$$$ second ar graph \n {gm.graph.print_tabular()}\n $$$$$$$$$$$$$$ \n"
    )

    _scatter_results_from_buffer(gi, gm, fe_list[:2])

    # final review print
    # graph_cleanup(gm)

    _debug(f" {pretty_print_graph(gm, 'final version, fusion pass')}")

    result = True  # TODO - make this mean something

    return gm
