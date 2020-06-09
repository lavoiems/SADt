import os
import random
import torch.utils.data as data
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def svhn(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose((
        transforms.ToTensor(),))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, valid_loader, test_loader, shape, n_classes


def svhn_extra(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose((
        transforms.ToTensor(),))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    extra = datasets.SVHN(root, split='extra', download=True, transform=transform)
    train = torch.utils.data.ConcatDataset((train, extra))
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, valid_loader, test_loader, shape, n_classes


def triple_channel(x):
    if x.shape[0] == 3:
        return x
    return torch.cat((x,x,x), 0)


def mnist(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, valid_loader, test_loader, shape, n_classes


def imnist(root, train_batch_size, test_batch_size, valid_split, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    n_classes = len(set(train.train_labels.tolist()))
    t = [transforms.RandomAffine(30, (0, 0), (0.5, 1.5), 40), transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=False, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, test_loader, shape, n_classes


def visda(root, train_batch_size, test_batch_size, use_normalize=False, **kwargs):
    train_transform = [
        transforms.Resize((256, 256), interpolation=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transform = [
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor(),
    ]

    if use_normalize:
        normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        train_transform.append(normalize)
        test_transform.append(normalize)

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    train = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, pin_memory=False,
                                               shuffle=True, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False,
                                              num_workers=10, drop_last=False)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))
    return train_loader, test_loader, shape, n_classes


def cond_visda(root1, root2, train_batch_size, test_batch_size, semantics, nc, device, **kwargs):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor(),
        normalize,
    ])

    train1 = datasets.ImageFolder(os.path.join(root1, 'train'), transform=train_transform)
    train2 = datasets.ImageFolder(os.path.join(root2, 'train'), transform=test_transform)
    train = CondDataset(train1, train2, semantics, nc, device)
    test1 = datasets.ImageFolder(os.path.join(root1, 'test'), transform=train_transform)
    test2 = datasets.ImageFolder(os.path.join(root2, 'test'), transform=test_transform)
    test = CondDataset(test1, test2, semantics, nc, device)

    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=True,
                                   num_workers=10, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(test, batch_size=test_batch_size, shuffle=False,
                                  num_workers=10, drop_last=False)
    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, nc


class CondDataset(data.Dataset):
    def __init__(self, dataset1, dataset2, semantics, nc, device):
        labels = []
        print('Infering semantics for dataset1')
        for sample, _ in dataset1:
            sample = sample.to(device)
            sample.unsqueeze_(0)
            sample = (sample+1)*0.5
            label = semantics(sample).argmax(1)
            labels.append(label)
        print('Infering semantics for dataset2')
        for sample, _ in dataset2:
            sample = sample.to(device)
            label = semantics((sample.unsqueeze(0) + 1) * 0.5).argmax(1)
            labels.append(label)

        self.labels = torch.LongTensor(labels)
        self.labels_idxs = [torch.nonzero(self.labels == label)[:, 0] for label in range(nc)]
        self.len_domain1 = len(dataset1)
        self.dataset = data.ConcatDataset((dataset1, dataset2))

    def __getitem__(self, idx):
        sample, _ = self.dataset[idx]
        target = self.labels[idx]
        domain = int(idx > self.len_domain1)
        idxs = self.labels_idxs[target]
        idx2 = idxs[random.randint(0, len(idxs)-1)]

        sample2, _ = self.dataset[idx2]
        target2 = self.labels[idx2]
        assert(target == target2)
        domain2 = int(idx2 > self.len_domain1)
        return sample, target, domain, sample2, domain2

    def __len__(self):
        return len(self.dataset)


class MultiTransformDataset(data.Dataset):
    def __init__(self, dataset, t):
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.ToPILImage()] +
            t)

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        return input, self.transform(input), target

    def __len__(self):
        return len(self.dataset)
