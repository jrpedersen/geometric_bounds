
from collections import OrderedDict
from functools import partial
import torch
from torch import nn, optim, linalg
from functorch import jacrev, vmap

def calc_bound(w, h, jh):
    return (
        (1 + linalg.vector_norm(h, dim=1)**2) /
        (linalg.matrix_norm(w, ord=2)**2 *
        linalg.matrix_norm(jh, ord='fro')**2)
    )

def calc_tight_bound(w, h, jh):
    return (
        (1 + linalg.vector_norm(h, dim=1)**2) /
        (linalg.matrix_norm(torch.bmm(w, jh), ord='fro')**2)
    )

def mlp_bound(weight_layer, partial_net, x, bound_f = calc_tight_bound):
    h_0 = x.view(x.shape[0],-1)
    if len(partial_net) > 0:
        h_n = partial_net(h_0)
        jh_n = vmap(jacrev(partial_net))(h_0)
    else:
        h_n = h_0
        jh_n = torch.eye(h_0.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1)

    w = weight_layer.weight.repeat(x.shape[0],1,1)
    return bound_f(w, h_n, jh_n), (w,h_n,jh_n)

def skip_mlp_bound(weight_layer, partial_net, x, bound_f = calc_tight_bound):
    negative_slope = weight_layer.activation.negative_slope

    h_0 = x.view(x.shape[0],-1)
    h_n = partial_net(h_0)
    jh_n = vmap(jacrev(partial_net))(h_0)

    w_base = weight_layer.fc.weight.repeat(x.shape[0],1,1)
    w_skip =torch.eye(*w_base.shape[1:]).unsqueeze(0).repeat(x.shape[0],1,1)
    w_skip = w_skip * (1 + (torch.gt(-weight_layer.fc(h_n), 0) * (negative_slope**(-1) - 1 ))).unsqueeze(1)
    w = w_base + w_skip
    return bound_f(w, h_n, jh_n), (w,h_n,jh_n)

def get_bounds(model, x):
    modules_list = list(model.layers._modules.items())
    bound_types = {
        'fc': mlp_bound,
        'sk': skip_mlp_bound
    }
    layer_wise_bound = []
    for i in range(len(modules_list)):
        name, layer = modules_list[i]
        if name[:2] in bound_types.keys():
            bound, _ = bound_types[name[:2]](
                layer,
                nn.Sequential(OrderedDict(modules_list[:i])),
                x
            )
            layer_wise_bound.append(bound)
    return torch.stack(layer_wise_bound).T.detach()

def compose_fns(f,g): return lambda x : g(f(x))

def norm_grad_params(model):
    return torch.tensor([(param.grad**2).sum() for param in model.parameters()])

def norm_grad_x(model, loss_fn, x, labels):
    grad_x = jacrev(compose_fns(model, partial(loss_fn,target=labels)))(x)
    return (grad_x**2).sum().detach()

def norm_grads(model, loss_fn, optimizer, x, labels):
    optimizer.zero_grad()
    inputs = x.clone()
    inputs.requires_grad_(True)
    inputs.retain_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()

    norm_grad_wrt_input = (inputs.grad**2).sum()
    norm_grad_wrt_params = torch.tensor([(param.grad**2).sum() for param in model.parameters()]).sum()
    return (norm_grad_wrt_input,
        norm_grad_wrt_params
    )
