import torch

def activation(x):
        return 1/(1+torch.exp(-x))

    torch.manual_seed(7)
    features = torch.randn((1,5))
    weights = torch.randn_like(features)
    bias = torch.randn((1,1))
    output = activation(torch.sum(features * weights) + bias)
    print(output)
    wei_matrix = weights.view(5,1)
    new_output = activation(torch.mm(features, wei_matrix) + bias)
    print(new_output)

    new_features = torch.randn((1,1))

    n_input = new_features.shape[1]
    n_hidden = 2
    n_output = 1
    W1 = torch.randn(n_input, n_hidden)
    W2 = torch.randn(n_hidden, n_output)

    B1 = torch.randn((1, n_hidden))
    B2 = torch.randn((1, n_output))

    h = activation(torch.mm(new_features, W1) + B1)
    output2 = activation(torch.mm(h, W2) + B2)

    print(output2)
