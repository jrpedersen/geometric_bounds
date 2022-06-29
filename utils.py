import torch
from torch import nn, optim, linalg

visualisation = {}
def named_hook(m,i,o, name=None):
    #print(type(i))
    #for elem in i: print(type(elem))
    #print(i)
    visualisation[name] = i

def hook_fn(m, i, o):
    visualisation[m] = i

def get_leaf_layers(m):
    children = list(m.children())
    if not children:
        return [m]
    leaves = []
    for l in children:
        leaves.extend(get_leaf_layers(l))
    return leaves


def get_all_layers(net):
    for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
        # it's a non sequential. Register a hook
            hook = partial(named_hook, name=name)
            layer.register_forward_hook(hook)
            # layer.register_forward_hook(hook_fn)


def get_hn_and_jhn(model, x):
    h_0 = x.view(1,-1)

    h_n = [h_0]
    jh_0 = torch.eye(h_0.shape[1]).unsqueeze(0)
    jh_n = [jh_0]

    for n in range(1,3):
        partial_net =nn.Sequential(OrderedDict(list(model.layers._modules.items())[:(n*2)]))
        h_n.append(partial_net(h_0))
        jh_n.append(torch.squeeze(jacrev(partial_net)(h_0)))
    return h_n, jh_n

def old_mlp_bound(model, x):
    h_n, jh_n = get_hn_and_jhn(model, x)
    params = [param for (i,param) in enumerate(model.parameters()) if i%2==0]
    bound = 0
    for w, h, jh in zip(params,h_n,jh_n):
        bound += (
            (1 + linalg.vector_norm(h)**2) /
            (linalg.matrix_norm(w, ord=2)**2 *
            linalg.matrix_norm(jh, ord='fro')**2)
        )
    return bound

def tighter_mlp_bound(model, x):
    h_n, jh_n = get_hn_and_jhn(model, x)
    params = [param for (i,param) in enumerate(model.parameters()) if i%2==0]
    bound = 0
    for w, h, jh in zip(params,h_n,jh_n):
        bound += (
            (1 + linalg.vector_norm(h)**2) /
            (linalg.matrix_norm(torch.matmul(w, jh), ord='fro')**2)
        )
    return bound
