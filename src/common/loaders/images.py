import math
import gzip
import random
import codecs
import torch.utils.data as data
from PIL import Image
import errno
import os
import os.path
import numpy as np
from torchvision.datasets.utils import download_url, makedir_exist_ok
from torch.utils.model_zoo import tqdm
import torch
from torchvision import datasets, transforms
from skimage import transform, filters
from quickdraw import QuickDrawData
from torch.utils.data.sampler import SubsetRandomSampler


class Rotate(object):
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        d = random.randrange(-30, 30)
        img = transform.rotate(img, d, mode='edge', order=4)
        return img


class Jitter(object):
    def __call__(self, img):
        if random.random() > 0.75:
            return img
        img = transforms.ColorJitter((0.5, 1), (0.5, 1), 1, 0.5)(img)
        return img


class Rescale(object):
    def __call__(self, img):
        img = transforms.RandomResizedCrop(img.size[1], scale=(0.5, 1.5))(img)
        return img


def svhn(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose((
        transforms.ToTensor(),))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    #extra = datasets.SVHN(root, split='extra', download=True, transform=transform)
    #train = torch.utils.data.ConcatDataset((train, extra))
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


def quickdraw(root, train_batch_size, test_batch_size, **kwargs):
    classes = ['t-shirt', 'pants', 'shoe', 'purse']
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=1),
        transforms.ToTensor(),
    ])
    train = QuickDrawDataset(root, classes, transform)
    test = QuickDrawDataset(root, classes, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))

    return train_loader, test_loader, shape, n_classes


def iquickdraw(root, train_batch_size, test_batch_size, **kwargs):
    classes = ['t-shirt', 'pants', 'shoe', 'purse']
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=1),
        transforms.ToTensor(),
    ])
    train = QuickDrawDataset(root, classes, transform)
    n_classes = len(set(train.classes))
    t = [
        transforms.RandomAffine(30, (0, 0), (0.8, 1.2), 30, fillcolor=(255,255,255)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = QuickDrawDataset(root, classes, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=False, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


def fashion(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)

    classes = [0, 1, 7, 8]
    train = FilterDataset(train, classes)
    test = FilterDataset(test, classes)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))
    return train_loader, test_loader, shape, n_classes


def inverse(x):
    return 1-x


def omniglot(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        inverse,
        triple_channel,
    ])
    train = datasets.ImageFolder(root, transform=transform)
    n_classes = 100
    t = [transforms.RandomAffine(30, (0, 0), (0.7, 1.3), 40), transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = datasets.ImageFolder(root, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


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


def isvhn(root, train_batch_size, test_batch_size, valid_split, **kwargs):
    transform = transforms.Compose((
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    shape = train[0][0].shape
    t = [Rescale(), Jitter(), Rotate(), transforms.ToTensor(), triple_channel]
    train = MultiTransformDataset(train, t)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    return train_loader, test_loader, test_loader, shape, n_classes


class FilterDataset(data.Dataset):
    def __init__(self, dataset, classes):
        filtered_dataset = list(filter(lambda x: x[1] in classes, dataset))
        images = list(map(lambda x: x[0], filtered_dataset))
        labels = [classes.index(data[1]) for data in filtered_dataset]

        self.tensor = images
        self.classes = labels

    def __getitem__(self, idx):
        input = self.tensor[idx]
        target = self.classes[idx]
        return input, target

    def __len__(self):
        return len(self.tensor)


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


class QuickDrawDataset(data.Dataset):
    def __init__(self, root, classes, transform):
        self.classes = classes
        self.labels = torch.arange(len(classes))
        self.transform = transform
        self.qdd = QuickDrawData(recognized=True, max_drawings=10000, cache_dir=root)
        self.qdd.load_drawings(classes)

    def __getitem__(self, idx):
        c = self.classes[idx%len(self.classes)]
        label = self.labels[idx%len(self.classes)]
        img = self.qdd.get_drawing(c).image
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return 10000


