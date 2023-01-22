import torch
import torch._dynamo as dynamo
from typing import Union, List, Literal

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b, img_size=224):
    return (
        torch.randn(b, 3, img_size, img_size).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

#def generate_data_linear(b ):
#    return 

N_ITERS = 10

from torchvision.models import resnet18

from torchvision.models import vit_b_16

import torch.nn as nn


class ReplicaModel(nn.Module):
    def __init__(self, layer_count: int = 2, _with_bias: bool = False) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(10, 10, bias=_with_bias) for _ in range(layer_count)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Literal[0]]:
        return sum([self.seq(x)])


# control depth of ReplicaModel
layers = 4
_device_type = "cuda" if torch.cuda.is_available() else "cpu"

# model = Permute().to(rank)  #
model = ReplicaModel(layer_count=layers).to(_device_type)

# permute_input = x = torch.randn(2, 10, 40).to("cuda")
x = torch.randn(2, 10).to(_device_type)


# fire off comms
# spmd(x).sum().backward()


def init_model(model):
    return model.to(torch.float32).cuda()


def eval(mod, inp):
    return mod(inp)


model = init_model()
# eval_opt = dynamo.optimize("inductor")(eval)


inp = generate_data(16)[0]
# print("eager:", timed(lambda: eval(model, inp))[1])
# print("dynamo:", timed(lambda: eval_opt(model, inp))[1])

"""eager_times = []
dynamo_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, eager_time = timed(lambda: eval(model, inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

dynamo_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, dynamo_time = timed(lambda: eval_opt(model, inp))
    dynamo_times.append(dynamo_time)
    print(f"dynamo eval time {i}: {dynamo_time}")
print("~" * 10)



eager_med = np.median(eager_times)
dynamo_med = np.median(dynamo_times)
speedup = eager_med / dynamo_med
print(
    f"(eval) eager median: {eager_med}, dynamo median: {dynamo_med}, speedup: {speedup}x"
)
print("~" * 10)

"""
import numpy as np

model_shell = ReplicaModel()

model = init_model(model_shell)
opt = torch.optim.Adam(model.parameters())


def train(mod, data):
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()


eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    opt.zero_grad(True)
    _, eager_time = timed(lambda: train(model, inp))
    opt.step()
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = dynamo.optimize("inductor")(train)

dynamo_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    opt.zero_grad(True)
    _, dynamo_time = timed(lambda: train_opt(model, inp))
    opt.step()
    dynamo_times.append(dynamo_time)
    print(f"dynamo train time {i}: {dynamo_time}")
print("~" * 10)

eager_med = np.median(eager_times)
dynamo_med = np.median(dynamo_times)
speedup = eager_med / dynamo_med
print(
    f"(train) eager median: {eager_med}, dynamo median: {dynamo_med}, speedup: {speedup}x"
)
print("~" * 10)

print(f"custom")
from typing import List

"""
def custom_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


# Reset since we are using a different backend (a custom one).
dynamo.reset()
opt_model = dynamo.optimize(custom_backend)(init_model())
opt_model(generate_data(16)[0])


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = dynamo.optimize(custom_backend)(bar)
inp1 = torch.randn(10)
inp2 = torch.randn(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

(
    explanation,
    out_guards,
    graphs,
    ops_per_graph,
    break_reasons,
    explanation_verbose,
) = dynamo.explain(bar, torch.randn(10), torch.randn(10))
print(explanation_verbose)

import traceback as tb

try:
    dynamo.export(bar, torch.randn(10), torch.randn(10))
except:
    tb.print_exc()

model_exp = dynamo.export(init_model(), generate_data(16)[0])
print(f"{model_exp=}")
print(model_exp[0](generate_data(16)[0]))

"""
