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


def train_one_epoch(training_loader, model, loss_fn,optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # test bounds and gradients
        bound = get_bounds(model, inputs).sum()
        norm_grad_wrt_x = norm_grad_x(model, loss_fn, inputs, labels)
        norm_grad_wrt_params = norm_grad_params(model).sum()
        print('Gradient with x:   ', norm_grad_wrt_x)
        print('Bound:             ', bound)
        print('Gradient with p:   ', norm_grad_wrt_params)
        print(f"Is {bound*norm_grad_wrt_x} < {norm_grad_wrt_params}")
        print(bound*norm_grad_wrt_x < norm_grad_wrt_params)
        pdb.set_trace()
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def main_manual_train():
    mnist = MNISTDataModule(data_dir="./data", batch_size=1)
    weakly_relu_neg_slope = 0.01
    #simple_mlp = SimpleMLP([40,40,10], 0.01)
    #simple_mlp = SimpleMLP([40,40,40,40,10], 0.01)
    model = SkipMLP([40,40,40,40,10], weakly_relu_neg_slope)
    model_name = re.findall(r"[\w]+", str(type(model)))[-1]
    logger = TensorBoardLogger('lightning_logs/', name=model_name)
    mnist.setup()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_one_epoch(mnist.train_dataloader(), model, nn.CrossEntropyLoss(label_smoothing=0.1), optimizer, 1, None)
