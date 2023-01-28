import os
import torch
import logging
from typing import Any, List, Literal, Union
import random
import torch.multiprocessing as mp

import torch.distributed as dist
from functools import partial
from torch._dynamo.utils import same

from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.distributed.distributed_c10d import _get_default_group
from torch._C._distributed_c10d import _register_process_group
from torch._dispatch.python import enable_python_dispatcher

import torch.nn as nn

import torch.distributed.traceable_collectives


def rank0_debug(logger: logging.Logger, *args: Any, **kwargs: Any) -> None:
    # print(f"debug - {args=}")
    if dist.is_initialized() and dist.get_rank() == 0:
        # print(f"{args}")  # , {logger=}")
        logger.debug(*args, **kwargs)


logger: logging.Logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logging.basicConfig(level="DEBUG")
_debug = partial(rank0_debug, logger)  # type: ignore


def setup(rank: int, world_size: int, use_cuda: bool = True) -> None:
    # logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)

    if use_cuda:

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        _debug("--> init process group using nccl")
        torch.cuda.set_device(rank)
        print(f"--> device set for rank {rank}")
    else:

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        _debug("--> init process group using gloo")

    # torch._inductor.config.debug = True


def teardown(rank: int) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    _debug(f"rank {rank} entering teardown")
    dist.barrier()
    _debug(f"shut down process group on rank {rank}")
    dist.destroy_process_group()


class ReplicaModel(nn.Module):
    def __init__(self, layer_count: int = 2, _with_bias: bool = False) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(10, 10, bias=_with_bias) for _ in range(layer_count)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Literal[0]]:
        return sum([self.seq(x)])


def main(rank: int, world_size: int, use_cuda: bool = True) -> None:

    # init
    setup(rank, world_size, use_cuda)
    print(f"71")
    _world_size = dist.get_world_size()
    _debug(f"--> World size = {_world_size}")

    # main work
    work_main(rank, world_size)

    # teardown
    teardown(rank)


def matmul_cat_col(a, b, c, d, e, f, *, pg_id):
    x = torch.matmul(a, b)
    y = torch.matmul(c, d)
    z = torch.cat((x, y))
    _debug(f"{z.shape=}")
    ar = torch.ops.aten.all_reduce(z, group_id=pg_id, reduce_op="sum")
    g = torch.matmul(e, f)
    out = torch.add(ar, g.repeat(2, 1))
    return (out,)


def compile(func, example_inputs):
    gm = make_fx(func)(*example_inputs)
    _debug(f"======== make fx ===============\n")
    _debug(f"{gm.graph.print_tabular()}\n")
    # print(f"{graph=}")
    gm_inductor = inductor_compile_fx(gm, example_inputs)
    _debug(f"======== inductor fx ===============\n")
    _debug(f"{gm_inductor=}\n")

    return gm_inductor


def work_main(rank: int, world_size: int) -> None:
    # must randomize the seed per GPU to ensure all_reduce
    # is meaningful.
    torch.manual_seed(rank)

    _device_type = "cuda" if torch.cuda.is_available() else "cpu"

    global matmul_cat_col

    pg = _get_default_group()
    _debug(f"{pg=}")
    pg_id = _register_process_group(pg)
    _debug(f"{pg_id=}")
    matmul_cat_col = partial(matmul_cat_col, pg_id=pg_id)
    inputs = (torch.ones(4, 4, device="cuda") + rank,) * 6

    with enable_python_dispatcher():
        correct_out = matmul_cat_col(*inputs)
        _debug(f"{correct_out=}")
        compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
        _debug(f"{compiled_matmul_cat_col=}")
        inductor_out = compiled_matmul_cat_col(*inputs)
        _debug(f"{inductor_out=}")

        _debug(f"rank {rank}: {correct_out}, {inductor_out}")
        assert same(correct_out, inductor_out)


if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = "localhost"
    # obtain random port
    port = random.randint(49152, 65535)
    os.environ["MASTER_PORT"] = str(port)

    world_size = 2
    assert torch.cuda.is_available(), "GPUs are needed to run this example!"

    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
