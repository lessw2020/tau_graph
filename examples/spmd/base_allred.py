import os
import torch
import logging
from typing import Any

print(f" torch version = {torch.__version__}")
import torch.distributed as dist
from functools import partial
from torch._dynamo.utils import same

from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.distributed.distributed_c10d import _get_default_group
from torch._C._distributed_c10d import _register_process_group
from torch._dispatch.python import enable_python_dispatcher

import torch.distributed.traceable_collectives

def rank0_debug(logger: logging.Logger, *args: Any, **kwargs: Any) -> None:
    #if dist.is_initialized() and 
    if dist.get_rank() == 0:
        logger.debug(*args, **kwargs)

logger: logging.Logger = logging.getLogger(__name__)
_debug = partial(rank0_debug, logger)  # type: ignore




def matmul_cat_col(a, b, c, d, e, f, *, pg_id):
    x = torch.matmul(a, b)
    y = torch.matmul(c, d)
    z = torch.cat((x, y))
    z1 = torch.cat((y,x))
    x1 = torch.matmul(a,b)
    #print(f"{z=}")
    ar = torch.ops.aten.all_reduce(z, group_id=pg_id, reduce_op="sum")
    g = torch.matmul(e, f)
    gz = torch.cat((g,z))
    ar2 = torch.ops.aten.all_reduce(z1, group_id=pg_id, reduce_op="sum")
    full_out = torch.add(ar2, ar)
    #out = torch.add(ar, g.repeat(2, 1))
    return (full_out,)


def compile(func, example_inputs):
    graph = make_fx(func)(*example_inputs)
    print(f"type of graph {type(graph)}")
    #print(f"{graph=}")
    return inductor_compile_fx(graph, example_inputs)


if __name__ == "__main__":
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "4")
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12344")
    rank = int(os.getenv("RANK"))
    dist.init_process_group(backend="nccl")
    _debug(f"{rank=}")
    world_size = int(os.getenv("WORLD_SIZE"))
    _debug(f"{world_size=}")
    torch.cuda.set_device(rank)
    

    torch._inductor.config.debug = True

    # this is a useless thing to do for the simple case of using default pg.
    # however, i did it to demonstrate the API proposed, whereby pg as int is passed
    # to collective APIs and pg object is recovered in execution layer
    pg = _get_default_group()
    _debug(f"{pg=}")
    pg_id = _register_process_group(pg)
    _debug(f"{pg_id=}")
    matmul_cat_col = partial(matmul_cat_col, pg_id=pg_id)
    inputs = (torch.ones(4, 4, device="cuda") + rank,) * 6
    #print(f"{inputs=}")

    with enable_python_dispatcher():
        correct_out = matmul_cat_col(*inputs)
        _debug(f"{correct_out=}")
        compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
        _debug(f"{compiled_matmul_cat_col=}")
        inductor_out = compiled_matmul_cat_col(*inputs)
        _debug(f"{inductor_out=}")

        _debug(f"rank {rank}: {correct_out}, {inductor_out}")
        assert same(correct_out, inductor_out)




"""def setup(rank: int, world_size: int, use_cuda: bool = True) -> None:
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)

    if use_cuda:
        _debug("--> init process group using nccl")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"--> device set for rank {rank}")
    else:
        _debug("--> init process group using gloo")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def teardown(rank: int) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    _debug(f"rank {rank} entering teardown")
    dist.barrier()
    dist.destroy_process_group()
    _debug(f"shut down process group on rank {rank}")


def main(rank: int, world_size: int, use_cuda: bool = True) -> None:

    # init
    setup(rank, world_size, use_cuda)

    _world_size = dist.get_world_size()
    logging.info(f"--> World size = {_world_size}")

    # main work
    work_main(rank, world_size)

    # teardown
    teardown(rank)


if __name__ == "__main__":
    import random

    os.environ["MASTER_ADDR"] = "localhost"
    # obtain random port
    port = random.randint(49152, 65535)
    os.environ["MASTER_PORT"] = str(port)

    world_size = 2
    # use_cuda: bool = DEVICE_TYPE == "cuda"

    # print(f"use_cuda == {use_cuda}, starting run_SPMD...\n")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    """
    
