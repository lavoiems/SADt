from importlib import import_module
from .train import train
from common.loaders import images
from common.util import get_args
from common.initialize import load_last_model


def parse_args(parser):
    parser.add_argument('--semantic-model-path', type=str, required=True)
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
    parser.add_argument('--dataset1', type=str, default='visda')
    parser.add_argument('--dataset2', type=str, default='visda')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--radius', type=float, default=3.5)
    parser.add_argument('--gsxy', type=float, default=1)
    parser.add_argument('--nc', type=float, default=1)


def execute(args):
    dataset1 = getattr(images, args.dataset1)
    dataset2 = getattr(images, args.dataset2)
    train_loader1, valid_loader1, test_loader1, shape1, nc = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size, args.valid_split)
    train_loader2, valid_loader2, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size, args.valid_split)
    args.loaders1 = (train_loader1, valid_loader1, test_loader1)
    args.loaders2 = (train_loader2, valid_loader2, test_loader2)
    args.shape1 = shape1
    args.shape2 = shape2

    model_definition = import_module('.'.join(('models', 'vmtc_repr', 'train')))
    model_parameters = get_args(args.semantic_model_path)
    model_parameters['nc'] = nc
    models = model_definition.define_models(shape1, **model_parameters)
    semantic = models['classifier']
    semantic = load_last_model(semantic, 'classifier', args.semantic_model_path)
    args.semantic = semantic

    train(args)
    evaluate_fid(args)
