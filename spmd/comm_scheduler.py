import collections
import dataclasses
import functools
import itertools
import logging
import os
import pprint
import textwrap
from typing import Dict, List, Optional, Set

import sympy

import torch
from torch._dynamo.utils import dynamo_timed

from . import config, dependencies, ir, metrics
from .dependencies import StarDep
from .sizevars import SimplifyIndexing
from .utils import cache_on_self, cmp, has_triton
from .virtualized import V

import torch.distributed as dist

log = logging.getLogger(__name__)

# from .debug import create_fx_from_snodes, draw_buffers

# from .debug import draw_buffers


def pformat(obj):
    if isinstance(obj, set):
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' '*4)}"
    return result


class OutputNode:
    def __init__(self, dep):
        self.unmet_dependencies = {dep}
        self.inverse_users = []

    def is_reduction(self):
        return False

    def get_alias_names(self):
        return ()

    def get_name(self):
        return "OUTPUT"

    __repr__ = get_name


class BaseSchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.Buffer):
        self.scheduler: "Scheduler" = scheduler
        self.node: ir.Buffer = node
        self.users: Optional[List[NodeUser]] = None
        self.inverse_users: List[BaseSchedulerNode] = []
        self.set_read_writes(node.get_read_writes())
        self.recursive_predecessors: Optional[Set[str]] = None
        self.min_order: Optional[int] = None
        self.max_order: Optional[int] = None
        self.last_usage: Set[str] = None  # buffers that won't be used after this kernel
        self.written = False

    def __repr__(self):
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self):
        """Longer form printout for trace logs"""
        name = self.get_name()
        lines = [
            f"{name}: {type(self).__name__}({type(self.node).__name__})",
            f"{name}.writes = {pformat(self.read_writes.writes)}",
            f"{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}",
            f"{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}",
        ]
        try:
            lines += [
                self.debug_str_extra(),
            ]
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)
        return "\n".join(lines).rstrip()

    def debug_str_extra(self):
        return ""

    def log_details(self):
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def update_mutated_names(self, renames: Dict[str, str]):
        self.set_read_writes(self.read_writes.rename(renames))

    def add_mutation_dep(self, name):
        self.set_read_writes(self.read_writes.with_read(name))

    def set_users(self, users: List["NodeUser"]):
        # deduplicate
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = NodeUser(
                    use.node, result[id(use.node)].can_inplace and use.can_inplace
                )
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self):
        return self.node.get_alias_names()

    def get_mutations(self):
        return self.node.get_mutation_names()

    def has_aliasing_or_mutation(self):
        return bool(self.get_aliases() or self.get_mutations())

    def set_read_writes(self, rw: dependencies.ReadWrites):
        self.read_writes: dependencies.ReadWrites = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def used_buffer_names(self) -> Set[str]:
        return {
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        }

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def get_name(self) -> str:
        return self.node.get_name()

    def get_first_name(self) -> str:
        return self.get_name()

    def get_names(self) -> Set[str]:
        return set([self.get_name()])

    def get_nodes(self) -> List["BaseSchedulerNode"]:
        return [self]

    def get_device(self):
        return self.node.get_device()

    def is_reduction(self):
        return False

    def is_template(self):
        return False

    def is_extern(self):
        return False

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def allocate(self):
        if self.node.should_allocate():
            # if self.node should allocate or
            # if self.node is generated by TritonKernelTemplates
            # because Triton kernel could not allocate tensor itself
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def codegen_originating_info(self, buffer, only_once=True):
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        origins = self.node.origins
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are boring and samey
                continue

            out_lines.append("")
            # TODO(voz): Should the pragma be constant somewhere?
            out_lines.append("#pragma CMT ORIGIN:")
            out_lines.append(f"#pragma CMT {o.op} {o.target}")
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.split("|")[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        buffer.writelines(out_lines)
        self.written = True


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def debug_str_extra(self):
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'kernel', None)}"

    def is_extern(self):
        return True

    def allocate(self):
        # TODO copied from SchedulerNode.allocate() -- should find a way to share code instead.
        #   should this be moved to BaseSchedulerNode, or would that break something?

        # what does V.kernel mutations mean/imply?
        # if config.inplace_buffers and getattr(V.kernel, "mutations", None) is not None:
        if config.inplace_buffers:
            from .codegen.wrapper import buffer_reuse_key

            ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)
            for read in ordered_reads:
                input_node: BaseSchedulerNode = self.scheduler.name_to_node.get(
                    read.name
                )
                if input_node and V.graph.wrapper_code.can_reuse(input_node):
                    remaining_uses = [
                        x
                        for x in input_node.users
                        if x.node.get_name()
                        not in self.scheduler.available_buffer_names
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and not isinstance(
                            input_node.node.get_layout(),
                            (ir.MultiOutputLayout, ir.MutationLayout, ir.AliasedLayout),
                        )
                        and buffer_reuse_key(input_node.node)
                        == buffer_reuse_key(self.node)
                    ):
                        V.graph.wrapper_code.codegen_inplace_reuse(
                            input_node.node, self.node
                        )
                        # TODO - i skipped steps relating to mutation tracking here (compare to SchedulerNode.allocate).
                        # what am I missing this way?
                        return
        super().allocate()

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False

        if read_dep.name not in self.scheduler.name_to_node:
            # don't allow reuse of an 'input' buffer, we don't own it
            # (would this have been fixed if I tracked mutations properly above?)
            return False

        if not isinstance(self.node, torch._inductor.ir.AllReduce):
            # TODO make this a property of the IR
            return False

        if len(self.read_writes.writes) == 1:
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.numbytes_hint() == write_dep.numbytes_hint()

        return False


class NopKernelSchedulerNode(BaseSchedulerNode):
    pass


class SchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group_fn):
        super().__init__(scheduler, node)
        (
            self._sizes,
            self._body,
        ) = node.simplify_and_reorder()

        self.group = (node.get_device(), group_fn(self._sizes))

        if self.is_template():
            self.set_read_writes(node.normalized_read_writes())
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=True
                )
            )

        if self.is_reduction():
            # reduction has last (reduced) dim in its sizes, and some
            # downstream dependencies get confused by it
            self.read_writes.writes = self.read_writes.writes | {
                w.strip_last_size() for w in self.read_writes.writes
            }
            # reduction not on the last dim swaps the sizes, and downstream
            # dependencies expect unswapped
            # TODO swapping sizes doesn't work, leads to
            # File "/scratch/ngimel/work/repos/torchdynamo/torchinductor/sizevars.py", line 130, in guard_equals
            # if len(right.free_symbols) < len(left.free_symbols):
            # AttributeError: 'int' object has no attribute 'free_symbols'
            # even though memory dep looks correct
            # self.read_writes.writes = self.read_writes.writes | {
            #     w.maybe_swap_sizes() for w in self.read_writes.writes
            # }

    def debug_str_extra(self):
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        if self.get_aliases():
            lines.append(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            lines.append(f"{name}.mutations = {pformat(self.get_mutations())}")
        if isinstance(self._body, ir.LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))
        return "\n".join(lines)

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        return bool(self.node.get_reduction_type())

    def is_template(self):
        return isinstance(self.node, ir.TemplateBuffer)

    def allocate(self):
        if (
            not self.node.should_allocate()
            or self.node.get_alias_names()
            or self.node.get_mutation_names()
        ):
            return super().allocate()

        if config.inplace_buffers and getattr(V.kernel, "mutations", None) is not None:
            from .codegen.wrapper import buffer_reuse_key

            ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)
            for read in ordered_reads:
                input_node: BaseSchedulerNode = self.scheduler.name_to_node.get(
                    read.name
                )
                if input_node and V.graph.wrapper_code.can_reuse(input_node):
                    remaining_uses = [
                        x
                        for x in input_node.users
                        if x.node.get_name()
                        not in self.scheduler.available_buffer_names
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and not isinstance(
                            input_node.node.get_layout(),
                            (ir.MultiOutputLayout, ir.MutationLayout, ir.AliasedLayout),
                        )
                        and buffer_reuse_key(input_node.node)
                        == buffer_reuse_key(self.node)
                    ):
                        V.graph.wrapper_code.codegen_inplace_reuse(
                            input_node.node, self.node
                        )
                        V.kernel.args.make_inplace(
                            input_node.get_name(), self.get_name()
                        )
                        if isinstance(
                            V.kernel, torch._inductor.codegen.triton.TritonKernel
                        ):
                            V.kernel.mutations.add(input_node.get_name())
                            V.kernel.mutations.add(self.get_name())
                        return
        super().allocate()

    def run(self, *index_vars):
        self.mark_run()
        self.codegen(index_vars)

    def mark_run(self):
        self.allocate()

    def ranges_from_index_vars(self, index_vars):
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars):
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_read_writes(self):
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes

        def fn(index):
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])

        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if len(self.read_writes.writes) == 1 and hasattr(read_dep, "index"):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False


class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        assert node1.scheduler is node2.scheduler
        return cls(node1.scheduler, node1.get_nodes() + node2.get_nodes())

    def __init__(self, scheduler: "Scheduler", snodes: List[SchedulerNode]):
        # NB: No need to call super().__init__() because we don't need to re-use any of its logic.
        self.snodes = snodes
        self.scheduler = scheduler
        self.node = None  # type: ignore[assignment]
        self.users = None
        self.inverse_users = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group
        self.recursive_predecessors = functools.reduce(
            set.union, [x.recursive_predecessors for x in snodes]
        )
        self.set_read_writes(
            functools.reduce(
                dependencies.ReadWrites.merge, [x.read_writes for x in snodes]
            )
        )
        names = set(self.get_names())
        self.unmet_dependencies = {
            dep
            for dep in functools.reduce(
                set.union, [x.unmet_dependencies for x in snodes]
            )
            if dep.name not in names
        } - self.read_writes.writes
        self.min_order = min([x.min_order for x in self.snodes])
        self.max_order = max([x.max_order for x in self.snodes])

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_names(self) -> Set[str]:
        return functools.reduce(set.union, [x.get_names() for x in self.snodes])

    def debug_str_extra(self):
        return (
            f"{self.get_name()}.snodes = {pformat([x.get_name() for x in self.snodes])}"
        )

    @cache_on_self
    def used_buffer_names(self) -> Set[str]:
        return functools.reduce(set.union, [x.used_buffer_names() for x in self.snodes])

    def get_nodes(self) -> List[BaseSchedulerNode]:
        return self.snodes

    def __repr__(self):
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self):
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_template(self):
        return any(x.is_template() for x in self.snodes)

    def get_device(self):
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self):
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: Dict[str, str]):
        raise NotImplementedError

    def add_mutation_dep(self, name):
        raise NotImplementedError

    def set_users(self, users: List["NodeUser"]):
        raise NotImplementedError

    def get_aliases(self):
        raise NotImplementedError

    def get_mutations(self):
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        raise NotImplementedError

    def allocate(self):
        raise NotImplementedError

    def can_free(self):
        raise NotImplementedError


def pick_loop_order(stride_lengths, sizes, priority_idx=()):
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a, b):
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        stride_len_a = [sl[a] for sl in stride_lengths]
        stride_len_b = [sl[b] for sl in stride_lengths]

        # equivalent to
        # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
        a_first = all(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = all(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first and not b_first:
            return -1
        if b_first and not a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # if we have priority node, only use that node's order
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


@dataclasses.dataclass
class NodeUser:
    node: BaseSchedulerNode
    can_inplace: bool = False

    def get_name(self):
        return self.node.get_name()


class Scheduler:
    @dynamo_timed
    def __init__(self, nodes):
        super(Scheduler, self).__init__()
        self.backends = {}
        self.debugger = dist.get_rank() == 0

        self.nodes = []
        self.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }

        # comm specific
        self.enforce_pools = False
        self.has_comms = False
        self.ar_map = {}  # buf name -> node, index
        self.wait_map = {}  # buf name -> node, index
        self.comm_pools = []  # store boundaries of available nodes for op fusions
        metrics.num_possible_fusions = 0
        metrics.num_blocked_fusions = 0

        # fusion specific
        self.prefusion_len = 0
        self.postfusion_len = 0

        # convert ir to scheduler nodes
        for index, node in enumerate(nodes):
            assert (
                node.origins is not None
            ), "All nodes passed to scheduling must have an origin"
            if node.is_no_op():
                self.nodes.append(NopKernelSchedulerNode(self, node))
            elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
                group_fn = self.get_backend(node.get_device()).group_fn
                self.nodes.append(SchedulerNode(self, node, group_fn))
            elif isinstance(node, ir.ExternKernel):
                new_snode = ExternKernelSchedulerNode(self, node)
                # flag ar and waits
                if isinstance(node, ir.AllReduce):
                    self.ar_map[new_snode.get_name()] = (new_snode, node, index)
                elif isinstance(node, ir.Wait):
                    self.wait_map[new_snode.get_name()] = (new_snode, node, index)

                self.nodes.append(new_snode)
            else:
                raise NotImplementedError(node)
        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()
        # verify we found some ars
        if self.debugger:
            print(f"======= allreduce and wait mappings ==========\n")
            for k, v in self.ar_map.items():

                print(f"{k},\n {v[0]}, {type(v[1])}\n")
            for k, v in self.wait_map.items():

                print(f"{k},\n {v[0]}, {type(v[1])}\n")
            print(f"----------------------------\n")

        if len(self.ar_map):
            self.has_comms = True
            # compute pools
            self.comm_pools.append(-1)
            for k, v in self.ar_map.items():
                index = v[-1]
                self.comm_pools.append(index)
                name = v[1].get_name()
                # if self.debugger:
                #    print(f"adding {name} as boundary at index {index}\n")
            self.comm_pools.append(float("inf"))

            if self.debugger:
                print(f"comms: allreduces found in graph")
                print(f" comm pool set at {self.comm_pools}")

        self.name_to_node = {node.get_name(): node for node in self.nodes}
        self.name_to_fused_node = None  # set in fuse_nods()

        # we handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        self.mutation_real_name = {}
        # mutation_real_name: maps back to the original name for codegen
        self.mutation_renames = {}

        self.compute_dependencies()
        self.topological_sort_schedule()
        self.compute_predecessors()
        self.dead_node_elimination()

        # if self.debugger:
        #    self.debug_draw_graph(print_graph=True, fname="prefusion")

        if self.debugger:
            print(f"dne = {V.graph.removed_buffers=}")
        for k in self.wait_map.keys():
            assert k not in V.graph.removed_buffers, f"{k} was removed during DNE"

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        V.debug.ir_pre_fusion(self.nodes)

        self.num_orig_nodes = len(self.nodes)
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}

        prefusion_gm = self.debug_get_fx_gm()

        # compare snodes vs fx
        pre_count, pre_snodes = self.debug_show_snodes()

        self.prefusion_len = len(self.nodes)

        # --------- fusion -----------
        self.fuse_nodes()

        self.compute_last_usage()

        # -------- end fusion ---------
        self.postfusion_len = len(self.nodes)

        V.debug.ir_post_fusion(self.nodes)
        V.debug.graph_diagram(self.nodes)
        # self.debug_draw_graph()
        # if self.debugger:
        #    self.debug_draw_graph(print_graph=True, fname="post-fusion")

        # used during codegen:
        self.current_device = None
        self.buffer_names_to_free = set()
        self.buffer_names_no_longer_needed = set()

        # post fusion graph
        if self.debugger:
            post_count, post_snodes = self.debug_show_snodes()
            postfusion_gm = self.debug_get_fx_gm()

            # print(f"{prefusion_gm.graph.print_tabular()}")
            # print(f"============== divider ==================")
            # print(f"{postfusion_gm.graph.print_tabular()}")
            # print(f" =========== end fx graphs ==========")

            # extract data from graphs
            prebyte = self.debug_show_graph_data(prefusion_gm)
            postbyte = self.debug_show_graph_data(postfusion_gm)

            # compare
            print(f"{pre_count=}, {post_count=}")
            print(f"prefusion len = {len(prebyte)}, postfusion len = {len(postbyte)}")
            print(f"{prebyte=}\n {postbyte=}\n")

            # compare snodes
            # print(
            #    f"pre count = {len(pre_snodes)}, post fusion len = {len(post_snodes)}"
            # )
            print(
                f"---- stats ----\n {self.prefusion_len=}, \n{self.postfusion_len=}\n"
            )
            print(f"pre-snodes({len(pre_snodes)}): {pre_snodes}\n")
            print(f"post-snodes({len(post_snodes)}): {post_snodes}\n")

        # show stats for comm pools
        if self.debugger:
            print(
                f"total possible fusions = {metrics.num_all_possible_fusions},\n total blocked by comm {metrics.num_comm_blocked_fusions}"
            )
            if metrics.num_comm_blocked_fusions:
                pct_blocked = round(
                    metrics.num_comm_blocked_fusions / metrics.num_all_possible_fusions,
                    4,
                )
                print(f" pct blocked by comm pools: {pct_blocked}")

        # move all waits to end
        waits_moved = self.move_waits_to_end()

        if self.debugger:
            wmove_count, wmove_snodes = self.debug_show_snodes()
            print(f"moved {waits_moved} waits")
            print(f"after wait move = \n{wmove_snodes}")

        ## ---- end scheduler process ------

    def move_waits_to_end(self) -> int:
        orig_nodes = self.nodes[:]
        new_nodes = []
        wait_nodes = []
        for node in self.nodes:
            if not isinstance(node, ExternKernelSchedulerNode):
                new_nodes.append(node)
                continue
            name = node.get_name()
            # wait node
            if name in self.wait_map:
                wait_nodes.append(node)
                continue
            # other extern
            new_nodes.append(node)
        # confirm we got all the wait nodes
        assert len(wait_nodes) == len(
            self.wait_map
        ), f"failed to get all wait nodes during move to last {self.wait_map=} vs {wait_nodes=}"

        new_nodes.extend(wait_nodes)
        len_new = len(new_nodes)
        len_orig = len(orig_nodes)
        assert (
            len_new == len_orig
        ), f"failed to move all nodes during move waits to end {len_new=},{len_orig=}"

        self.nodes = new_nodes
        return len(wait_nodes)

    def debug_show_snodes(self, title=None):
        sequence = []
        # print(f"====== snodes graph {title} ============")
        for i, node in enumerate(self.nodes):
            total_size = 0
            name = node.get_name()
            if isinstance(node, FusedSchedulerNode):
                internal_nodes = node.get_nodes()

                for j, inode in enumerate(internal_nodes):
                    # print(f"{type(inode)}")
                    size = inode.node.get_size()
                    dtype = inode.node.get_dtype()
                    total_size += self.get_bytes(size, dtype)
                    # print(f"size {size}, dtype = {dtype}")
                # print(f"Fused Node has {total_size} bytes")
                out_string = "Fuse_" + str(j + 1) + name + "_nodes_" + str(total_size)
                sequence.append(out_string)

            elif isinstance(node, SchedulerNode):
                size = node.node.get_size()
                dtype = node.node.get_dtype()
                total_size = self.get_bytes(size, dtype)
                # print(f"size {size}, dtype = {dtype}")
                # print(f"{type(node)}")
                # print(f"{total_size=}")
                out_string = "sched_" + name + "_" + str(total_size)
                sequence.append(out_string)

            elif isinstance(node, ExternKernelSchedulerNode):

                size = node.node.get_size()
                dtype = node.node.get_dtype()
                total_size = self.get_bytes(size, dtype)

                if name in self.ar_map:
                    _, total_size, wraps = self.debug_inspect_comm(node)
                    out_string = "AR_" + name + "_" + str(total_size) + "_" + wraps

                elif name in self.wait_map:
                    _, total_size, wraps = self.debug_inspect_comm(node)
                    out_string = "Wait_" + name + "_" + str(total_size) + "_" + wraps
                    # print(f"wait processing...")

                else:
                    out_string = "Extern_" + name + "_" + str(total_size)

                sequence.append(out_string)

            elif isinstance(node, NopKernelSchedulerNode):
                size = node.node.get_size()
                dtype = node.node.get_dtype()
                total_size = self.get_bytes(size, dtype)
                # print(f"size {size}, dtype = {dtype}")
                # print(f"{type(node)}")
                # print(f"{total_size=}")
                out_string = "nop_" + name + "_" + str(total_size)
                sequence.append(out_string)

            else:
                print(f"unknown - {node.get_name()}, {type(node)}")
                sequence.append("Uknown")

        print(f"processed total of {i+1} nodes")
        # print(f" ------------------------\n")
        return i + 1, sequence

    def get_bytes(self, size, dtype):
        byte_mul = 1
        if dtype == torch.float32:
            byte_mul = 4
        elif dtype in [torch.bfloat16, torch.float16]:
            byte_mul = 2

        prod = 1
        for elem in size:
            prod *= elem
        prod *= byte_mul

        return prod

    def debug_inspect_comm(self, node):
        """queries a given scheduler comm node to determine size and dependency"""
        if not isinstance(node, ExternKernelSchedulerNode):
            raise ValueError(f"expected only comm related nodes, got {type(node)}")

        name = node.get_name()
        # print(f"{name=}")
        inner = node.node
        # print(f"{type(inner)}")
        size = inner.get_size()
        dtype = inner.get_dtype()
        total_size = self.get_bytes(size, dtype)
        inner_size = inner.get_numel()
        print(f"{inner_size=}, {size=}")
        print(f"{node=}")
        wraps = "?"
        if isinstance(inner, torch._inductor.ir.Wait):
            wraps = inner.inputs[0].name
        else:
            wraps = inner.inputs[0].data.name

        return name, total_size, wraps

    def debug_show_graph_data(self, gm) -> List:
        byte_schedule = []
        # do we have metadata
        postgraph = gm.graph
        for node in postgraph.nodes:
            name = node.name
            print(f"{name} ---- ")
            if node.op == "placeholder":
                continue
            if node.op == "output":
                byte_schedule.append("out")

            metadata = node.meta.get("tensor_meta", None)
            if metadata is None:
                continue

            print(f"metadata type = {type(metadata)}")
            # size = metadata.shape.numel()
            shape = metadata.shape
            if isinstance(shape, tuple):
                product = 1
                for elem in shape:
                    product *= elem
                print(f"{name} has shape {shape}, size {product*4} bytes")
                byte_schedule.append(product)
            else:
                if shape == "extern":
                    print(f"extern kernel")
                    if name in self.ar_map:
                        print(f"all-reduce node")
                        byte_schedule.append("AR")
                    elif name in self.wait_map:
                        print(f"wait node")
                        byte_schedule.append("Wait")
                print(f"{name} is type {shape}")

        # print(f"\n=== byte_schedule===\n {byte_schedule}")
        return byte_schedule

    def debug_get_fx_gm(
        self,
    ):
        """generate an fx graph module of snodes"""
        from .debug import get_fx_graphmodule

        gm = get_fx_graphmodule(self.nodes)
        return gm

    def debug_draw_graph(self, print_graph=True, fname="base"):
        """Generate an image of the graph for debugging"""
        # if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
        from .debug import draw_buffers

        assert False
        # draw_buffers(nodes, print_graph=False, fname=None):
        draw_buffers(self.nodes, print_graph=print_graph, fname=fname)

    def debug_print_nodes(self, label):
        if log.isEnabledFor(logging.INFO):
            log.info("%s:", label)
            for node in self.nodes:
                node.log_details()

    def compute_dependencies(self):
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """
        name_to_users = collections.defaultdict(list)

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node1 in self.nodes:
            node1_name = node1.get_name()
            for node2_name in node1.get_aliases():
                if node1_name in name_to_users and node2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[node1_name]
                    list2 = name_to_users[node2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if name_to_users[key] is list1 or name_to_users[key] is list2:
                            name_to_users[key] = combined
                elif node1_name in name_to_users:
                    name_to_users[node2_name] = name_to_users[node1_name]
                else:
                    name_to_users[node1_name] = name_to_users[node2_name]

        def rename(n):
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def dep_closure(node_name):
            reachable_names = {node_name}
            node = self.name_to_node[node_name]
            write_dep = list(node.read_writes.writes)[0]
            for read_dep in node.read_writes.reads:
                if (
                    read_dep.name in self.name_to_node
                    and read_dep.index == write_dep.index
                    and read_dep.size == write_dep.size
                ):
                    reachable_names.update(dep_closure(read_dep.name))
            return reachable_names

        def add_user(used_by_name, user_node, can_inplace=False):
            name_to_users[rename(used_by_name)].append(NodeUser(user_node, can_inplace))

        for node in self.nodes:
            # a node will mutate either 0 or 1 buffers
            for alt_name in node.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_mutation_dep(alt_name)
                for other_node in name_to_users[alt_name]:
                    # this node must run after all prior readers
                    other_name = rename(other_node.get_name())
                    known_dep_node_names = dep_closure(node.get_name())
                    if other_name not in known_dep_node_names:
                        # If this node alreay directly or indirectly depends on other_node,
                        # we don't need to insert an extra StarDep.
                        node.add_mutation_dep(other_name)
                        add_user(other_name, node)

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for alt_name in node.get_mutations():
                self.mutation_renames[rename(alt_name)] = node.get_name()
                self.mutation_renames[alt_name] = node.get_name()
                self.mutation_real_name[node.get_name()] = self.mutation_real_name.get(
                    alt_name, alt_name
                )

        # make sure outputs aren't dead-code-eliminated
        for node_name in V.graph.get_output_names():
            add_user(node_name, OutputNode(StarDep(node_name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)

        # copy users information onto the nodes
        for node in self.nodes:
            node.set_users(name_to_users[node.get_name()])

        # populate inverse_users
        for node in self.nodes:
            for user in node.users:
                user.node.inverse_users.append(node)

    def dead_node_elimination(self):
        updated_nodes = []
        for node in self.nodes:
            if node.users:
                updated_nodes.append(node)
            else:
                # dead code
                log.debug(f"removed node {node.get_name()}")
                V.graph.removed_buffers.add(node.get_name())
        self.nodes = updated_nodes

    def dead_node_elimination(self):
        """
        Remove any nodes without users
        """
        updated_nodes = []
        for node in self.nodes:
            if node.users:
                updated_nodes.append(node)
            else:
                # dead code
                log.debug("removed dead node: %s", node.get_name())
                V.graph.removed_buffers.add(node.get_name())
        self.nodes = updated_nodes

    def topological_sort_schedule(self):
        seen = set()
        name_to_node = {}
        result = []

        def visit(n):
            seen.add(n)
            # visit all dependencies
            for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                visit(name_to_node[dep.name])
            # at last step, save the final node
            result.append(n)

        # setup name_ to _node
        for node in self.nodes:
            for name in node.get_names():
                name_to_node[name] = node
        # visit all nodes
        for node in self.nodes:
            visit(node)
        # save topo sort as node graph
        self.nodes = result

    def topological_sort_schedule(self):

        """
        Ensure self.nodes is in topologically sorted order
        """
        seen = set()
        name_to_node = dict()
        result = []

        def visit(n):
            if n not in seen:
                seen.add(n)
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    visit(name_to_node[dep.name])
                result.append(n)

        for node in self.nodes:
            for name in node.get_names():
                name_to_node[name] = node
        for node in self.nodes:
            visit(node)
        self.nodes = result

    def compute_predecessors(self):
        """
        Populate each node.recursive_predecessors
        """
        # note self.nodes is topologically sorted
        name_to_predecessors = {}
        for node in self.nodes:
            recursive_predecessors = set()
            for dep in node.unmet_dependencies:
                recursive_predecessors.add(dep.name)
                recursive_predecessors |= name_to_predecessors[dep.name]
            name_to_predecessors[node.get_name()] = recursive_predecessors
            node.recursive_predecessors = recursive_predecessors

        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order

    def fuse_nodes(self):
        for _ in range(10):
            old_len = len(self.nodes)
            self.fuse_nodes_once()
            if len(self.nodes) == old_len:
                break

    def fuse_nodes(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.
        """
        for _ in range(10):
            old_len = len(self.nodes)
            self.fuse_nodes_once()
            if len(self.nodes) == old_len:
                break

    def fuse_nodes_once(self):
        fused_nodes = set(self.nodes)
        for n1, n2 in self.get_possible_fusions():
            node1 = self.name_to_fused_node[n1.get_first_name()]
            node2 = self.name_to_fused_node[n2.get_first_name()]
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                node3 = FusedSchedulerNode.fuse(node1, node2)
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.topological_sort_schedule()

    def fuse_nodes_once(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuses(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = set(self.nodes)
        for node1, node2 in self.get_possible_fusions():
            node1 = self.name_to_fused_node[node1.get_first_name()]
            node2 = self.name_to_fused_node[node2.get_first_name()]
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                node3 = FusedSchedulerNode.fuse(node1, node2)
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.topological_sort_schedule()

    def get_possible_fusions(self):
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen = set()
        # comm check
        def get_index(node1):
            name = node1.get_name()
            if "_" in name:
                index = node1.min_order
            else:
                index = int(name[3:])  # strip buf
            return index

        def get_bounds_linear_scan(index):
            # linear scan for now, binary search next
            low = -1
            high = float("inf")
            if self.debugger:
                print(f"start index = {index}")
            for ptr, upper in enumerate(self.comm_pools[1:]):
                # if self.debugger:
                # print(f"pool loop {index, ptr, upper}")
                if index > upper:
                    continue
                low = self.comm_pools[ptr]
                high = upper
            return (low, high)

        def get_bounds_binary_search(target):
            ars = self.comm_pools  # for shorter code
            l = 0
            r = len(self.comm_pools) - 1

            if self.debugger:
                print(f"bin search {target=} within {self.comm_pools}")

            # avoid bsearch if upper or lower extreme
            if target >= ars[-2]:
                if self.debugger:
                    print(f" short circuit for {target} via upper")
                return (ars[-2], ars[-1])
            if target < ars[1]:
                if self.debugger:
                    print(f" short circuit for {target} via lower")
                return (ars[0], ars[1])

            while l <= r:
                mid = l + (r - l) // 2
                midval = ars[mid]
                if self.debugger:
                    print(f"{mid=}, {midval=}, {l=}, {r=}")

                if target >= midval and target < ars[mid + 1]:
                    l = mid
                    r = mid + 1
                    if self.debugger:
                        print(f"located: {target=}, lower {ars[l]}, upper {ars[r]}")

                    break
                if target > midval:
                    l = mid + 1
                else:
                    r = mid

                assert (
                    ars[l] <= target < ars[r]
                ), f"target node index {target} is not within pool bounds {ars[l]} and {ars[r]}"

            return (ars[l], ars[r])  # low and high value

        def get_pool_bounds(index):
            """obtain min/max indexes for given comm pool"""
            # use binary search
            lowbin, highbin = get_bounds_binary_search(index)

            return (lowbin, highbin)

        def get_comm_bounds(node1):
            """get the pool for this node"""
            index = get_index(node1)
            lowval, highval = get_pool_bounds(index)
            return (lowval, highval)

        def is_valid_comm_pair(node2, low, high):
            index = get_index(node2)
            is_valid = index >= low and index < high
            # if self.debugger:
            #    print(f"is valid result {is_valid}, {index=}, {low=}, {high=}")
            return is_valid

        def check_all_pairs(nodes):
            for node1_index, node1 in enumerate(nodes):
                if self.has_comms and self.enforce_pools:
                    low, high = get_comm_bounds(node1)

                for node2 in nodes[node1_index + 1 :]:
                    metrics.num_all_possible_fusions += 1
                    if self.has_comms and self.enforce_pools:
                        if not (is_valid_comm_pair(node2, low, high)):
                            # if self.debugger:
                            #    print(f"blocked {node2.get_name()}, {low=}, {high=}")
                            metrics.num_comm_blocked_fusions += 1
                            continue
                    key = (node1, node2)
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.can_fuse(node1, node2):
                        possible_fusions.append(key)
                    elif node2.is_template() and self.can_fuse(node2, node1):
                        # epilogue fusions are order dependent
                        possible_fusions.append((node2, node1))

        buffer_names_grouping = collections.defaultdict(list)
        for node in self.nodes:
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.aggressive_fusion:
            group_grouping = collections.defaultdict(list)
            for node in self.nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        return sorted(possible_fusions, key=self.score_fusion_key, reverse=True)

    def will_fusion_create_cycle(self, node1, node2):
        """Finds whether there's a path from src to dst caused indirectly by fusion"""

        def check(node):
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                return bool(combined_names & node.recursive_predecessors) or any(
                    check(self.name_to_fused_node[n])
                    for n in node.recursive_predecessors - combined_predecessors
                )
            return False

        visited = set()
        combined_names = node1.get_names() | node2.get_names()
        combined_predecessors = (
            node1.recursive_predecessors | node2.recursive_predecessors
        ) - combined_names
        return any(check(self.name_to_fused_node[n]) for n in combined_predecessors)

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """
        if node1 is node2:
            return False
        if (
            isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node1.is_template()
        ):
            return False
        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            return False
        if node2.get_names() & node1.recursive_predecessors:
            return False  # node2 must go before node1
        if node2.is_template():
            return False  # only epilogues
        if node1.is_template() and (
            node2.has_aliasing_or_mutation()
            or node2.is_reduction()
            or not config.epilogue_fusion
        ):
            return False

        device = node1.get_device()
        if device != node2.get_device():
            return False  # wrong device

        no_shared_data = self.score_fusion_memory(node1, node2) == 0
        if no_shared_data and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            return False  # heuristic not needed for correctness

        if len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size:
            return False  # heuristic not needed for correctness

        if node1.get_names() & node2.recursive_predecessors:
            # node2 depends on node1 outputs
            if not self.can_fuse_vertical(node1, node2):
                return False
            return self.get_backend(device).can_fuse_vertical(node1, node2)
        else:  # nodes don't depend on each other, but may have common reads
            return self.get_backend(device).can_fuse_horizontal(node1, node2)

    def can_fuse_vertical(self, node1, node2):
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.
        """
        node1_names = node1.get_names()
        computed_deps = set()
        for rd in node2.unmet_dependencies:
            for cd in node1.read_writes.writes:
                # StarDep doesn't match MemoryDep, different indices don't match
                # However, broadcasting sometimes strips dimensions, and if that's the case
                # we still can match unmet dep
                if (
                    rd.name == cd.name
                    and type(rd) == type(cd)
                    and rd.index == cd.index
                    and len(rd.size) >= len(cd.size)
                    and rd.size[: len(cd.size)] == cd.size
                ):
                    computed_deps.add(rd)

        remaining_deps = {dep.name for dep in node2.unmet_dependencies - computed_deps}
        if remaining_deps & node1_names:
            # MemoryDeps didn't match and read different locations of the same buffer.
            # Examples here include:
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            return False
        for name in remaining_deps:
            if node1_names & self.name_to_fused_node[name].recursive_predecessors:
                return False
        return True

    def score_fusion(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Assign a score (higher comes first) to the fusion of node1
        and node2.  When different fusions conflict with each other,
        this is the way we decide what order to run them in.

        Our current score is based on:
        - Estimate of the saved memory operations
        - Fusions closer together in original order
        """
        memory_score = self.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return (
            node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )

    def score_fusion_memory(self, node1, node2):
        """
        The first term in our fusion score that estimates number of saved memory operations.
        """
        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        return sum(dep.numbytes_hint() for dep in common_memory_deps)

    def score_fusion_key(self, nodes):
        """
        Shim for list.sort(key=...)
        """
        node1, node2 = nodes
        return self.score_fusion(node1, node2)

    def compute_last_usage(self):
        """
        Populate node.last_usage
        """

        future_used_buffers = set()
        for node_name in V.graph.get_output_names():
            future_used_buffers.add(node_name)

        for node in reversed(self.nodes):
            used_buffers = node.used_buffer_names()
            used_buffers = {self.mutation_real_name.get(k, k) for k in used_buffers}
            node.last_usage = used_buffers - future_used_buffers
            future_used_buffers.update(used_buffers)

    def free_buffers(self):
        """Free any buffers that are no longer needed"""
        for name in sorted(self.buffer_names_to_free - V.graph.removed_buffers):
            if name in self.name_to_node:
                node = self.name_to_node[name]
                if node.can_free():
                    V.graph.wrapper_code.codegen_free(node.node)
            elif name in V.graph.graph_inputs:
                storage = V.graph.graph_inputs[name].data
                assert storage.is_input_buffer()
                V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

    def remove_kernel_local_buffers(self):
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """
        for name in V.kernel.store_buffer_names & self.buffer_names_no_longer_needed:
            if (
                name not in V.kernel.must_keep_buffers
                and name not in V.kernel.args.input_buffers
                and name not in self.mutation_renames
                and name not in self.mutation_real_name
            ):
                # For inplace buffers subject to remove, we don't actually
                # remove them but put them in a dedicated set. This simplifies
                # the life cycle management of inplace buffers.
                # This set is used to
                # 1) avoid unnecessary store in DeferredLine.
                # 2) avoid alias var definitions in kernel.
                if name in V.kernel.args.inplace_buffers:
                    V.graph.inplaced_to_remove.add(name)
                else:
                    self.remove_buffer(name)

    def remove_buffer(self, name):
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        V.kernel.args.output_buffers[name] = "REMOVED"
        V.graph.removed_buffers.add(name)

    def flush(self):
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode):
        assert isinstance(scheduler_node, ExternKernelSchedulerNode)
        scheduler_node.allocate()
        node = scheduler_node.node
        node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    def create_backend(self, device: torch.device):
        assert (
            device.type != "cuda" or device.index is not None
        ), f"{device} should have been normalized in lowering"
        V.graph.device_types.add(device.type)
        if device.type == "cpu":
            from .codegen.cpp import CppScheduling

            return CppScheduling(self)
        else:
            if not has_triton():
                device_props = torch.cuda.get_device_properties(device)
                if device_props.major < 7:
                    raise RuntimeError(
                        f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
                    )
                else:
                    raise RuntimeError(
                        "Cannot find a working triton installation. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
                    )
            from .codegen.triton import TritonScheduling

            return TritonScheduling(self)

    def get_backend(self, device: torch.device):
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    @dynamo_timed
    def codegen(self):
        for node in self.nodes:
            self.buffer_names_no_longer_needed.update(node.last_usage)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if device.type == "cuda":
                        if self.current_device and self.current_device.type == "cuda":
                            V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_cuda_device_guard_enter(
                            device.index
                        )
                    elif self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                    self.current_device = device

            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                self.codegen_extern_call(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_nodes(node.get_nodes())
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.allocate()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_names())

        self.flush()
