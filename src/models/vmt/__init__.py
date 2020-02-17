# Model inspired form: https://github.com/xudonmao/VMT

from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
    parser.add_argument('--dataset1', type=str, default='mnist')
    parser.add_argument('--dataset2', type=str, default='svhn')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--radius', type=float, default=3.5)
    parser.add_argument('--cw', type=float, default=1)
    parser.add_argument('--tcw', type=float, default=0.1)
    parser.add_argument('--dw', type=float, default=0.01)
    parser.add_argument('--svw', type=float, default=1)
    parser.add_argument('--tvw', type=float, default=0.1)
    parser.add_argument('--smw', type=float, default=1)
    parser.add_argument('--tmw', type=float, default=0.1)


def execute(args):
    print(args)
    dataset1 = getattr(images, args.dataset1)
    dataset2 = getattr(images, args.dataset2)
    train_loader1, _, test_loader1, shape1, nc = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size, args.valid_split)
    train_loader2, _, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size, args.valid_split)
    args.loaders1 = (train_loader1, test_loader1)
    args.loaders2 = (train_loader2, test_loader2)
    args.shape1 = shape1
    args.shape2 = shape2
    args.nc = nc

    train(args)
