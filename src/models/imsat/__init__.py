from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset-loc', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='imnist')
    parser.add_argument('--h-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--d-updates', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--ld', type=float, default=1)


def execute(args):
    print(args)
    dataset1 = getattr(images, args.dataset1)
    train_loader1, _, test_loader1, shape1, n_classes = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size, args.valid_split)
    args.loaders1 = (train_loader1, test_loader1)
    args.shape1 = shape1

    args.n_classes = n_classes

    train(args)
