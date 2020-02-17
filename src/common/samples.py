from .util import one_hot_embedding
import torch


def cube(*shape, **kwargs):
    epsilon = torch.rand(shape[0])
    c = torch.diag(epsilon)
    idx = torch.randint(low=0, high=shape[1], size=(shape[0],))
    idx = one_hot_embedding(idx, shape[1])
    samples = torch.mm(c, idx)

    return torch.as_tensor(samples, **kwargs).reshape(*shape)


def sphere(*shape, **kwargs):
    return torch.randn(*shape, **kwargs)
