from importlib import import_module
from .train import train
from common.loaders import images
from common.util import get_args
from common.initialize import load_last_model


def parse_args(parser):
    parser.add_argument('--semantic-model-path', type=str, required=True)
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cond_visda')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--radius', type=float, default=3.5)
    parser.add_argument('--gsxy', type=float, default=1)
    parser.add_argument('--nc', type=float, default=5)


def execute(args):
    dataset = getattr(images, args.dataset)

    model_definition = import_module('.'.join(('models', 'vmtc_repr', 'train')))
    model_parameters = get_args(args.semantic_model_path)
    model_parameters['nc'] = args.nc
    models = model_definition.define_models(**model_parameters)
    semantics = models['classifier']
    semantics = load_last_model(semantics, 'classifier', args.semantic_model_path)
    semantics = semantics.to(args.device)
    semantics.eval()

    train_loader, test_loader, shape, _ = dataset(args.dataset_loc1, args.dataset_loc2, args.train_batch_size,
                                                  args.test_batch_size, semantics, args.nc,
                                                  args.device)
    args.loaders1 = (train_loader, test_loader)
    args.shape = shape

    train(args)
