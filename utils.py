
visualisation = {}
def named_hook(m,i,o, name=None):
    #print(type(i))
    #for elem in i: print(type(elem))
    #print(i)
    visualisation[name] = i

def hook_fn(m, i, o):
    visualisation[m] = i

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
