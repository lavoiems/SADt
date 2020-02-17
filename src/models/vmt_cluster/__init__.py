from importlib import import_module
from .train import train
from common.loaders import images
from common.util import get_args
from common.initialize import load_last_model


def parse_args(parser):
    parser.add_argument('--cluster-model', type=str, required=True)
    parser.add_argument('--cluster-model-path', type=str, required=True)
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset1', type=str, default='mnist')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
    parser.add_argument('--dataset2', type=str, default='svhn')
    parser.add_argument('--h-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=64)
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
    train_loader1, _, test_loader1, shape1, nc = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size)
    args.loaders1 = (train_loader1, test_loader1)
    args.shape1 = shape1
    args.nc = nc

    dataset2 = getattr(images, args.dataset2)
    train_loader2, _, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size)
    args.loaders2 = (train_loader2, test_loader2)
    args.shape2 = shape2

    model_definition = import_module('.'.join(('models', args.cluster_model, 'train')))
    model_parameters = get_args(args.cluster_model_path)
    model_parameters['n_classes'] = nc
    models = model_definition.define_models(shape1, **model_parameters)
    cluster = models['encoder']
    cluster = load_last_model(cluster, 'encoder', args.cluster_model_path)
    args.cluster = cluster


    train(args)
