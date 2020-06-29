import torch
import torchvision
from importlib import import_module
from .train import train
from common.loaders import images
from common.util import get_args
from common.initialize import load_last_model


def parse_args(parser):
    parser.add_argument('--da-path', type=str, required=True, help='Path of the pre-trained domain adaptation network (vtmc_repr)')
    parser.add_argument('--ss-path', type=str, required=True, help='Path of the pre-trained self-supervised model (MoCO-v2)')
    parser.add_argument('--dataset-loc1', type=str, default='./data/sketch', help='Location of the first dataset')
    parser.add_argument('--dataset-loc2', type=str, default='./data/real', help='Location of the second dataset')
    parser.add_argument('--dataset', type=str, default='cond_visda', choice=['cond_visda', 'cond_mnist_svhn'], help='Dataset framework for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for Mapping Network')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay factor')
    parser.add_argument('--beta1', type=float, default=0, help='Adam parameter')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam parameter')
    parser.add_argument('--z-dim', type=int, default=16, help='N channels of the bottleneck')
    parser.add_argument('--style-dim', type=int, default=64, help='N channels of the style output')
    parser.add_argument('--max-conv-dim', type=int, default=512, help='Maximum N channels in the network')
    parser.add_argument('--bottleneck-size', type=int, default=64, help='Spatial dimension of the bottleneck')
    parser.add_argument('--n-unshared-layers', type=int, default=0, help='Number of unshared layer in the style encoder')
    parser.add_argument('--nc', type=float, default=5, help='Number of classes to transfer')
    parser.add_argument('--lambda_gp', type=float, default=1, help='Lambda gradient penalty')
    parser.add_argument('--lambda_dclass', type=float, default=1, help='Lambda classifier discriminator')
    parser.add_argument('--lambda_lcl', type=float, default=1, help='Lambda classifier loss generator')
    parser.add_argument('--lambda_lsty', type=float, default=1, help='Lambda style loss')
    parser.add_argument('--lambda_lcyc', type=float, default=1, help='Lambda cycle loss')


def semantics_fn(ss, da):
    def evaluate(x):
        o = ss(x)
        return da(o)
    return evaluate


def execute(args):
    dataset = getattr(images, args.dataset)

    model_definition = import_module('.'.join(('models', args.da_model, 'train')))
    model_parameters = get_args(args.da_path)
    model_parameters['nc'] = args.nc
    model_parameters['shape1'] = [3, args.img_dim]
    models = model_definition.define_models(**model_parameters)
    da = models['classifier']
    da = load_last_model(da, 'classifier', args.da_path)
    da = da.to(args.device)
    da.eval()

    if args.ss_path:
        ssx = torchvision.models.resnet50().to(args.device)
        ssx.fc = torch.nn.Identity()
        state_dict = torch.load(args.ss_path, map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        err = ssx.load_state_dict(state_dict, strict=False)
        print(err)
        ssx = ssx.to(args.device)
        ssx.eval()
        semantics = semantics_fn(ssx, da)
    else:
        semantics = da

    train_loader, test_loader, shape, _ = dataset(args.dataset_loc1, args.dataset_loc2, args.train_batch_size,
                                                  args.test_batch_size, semantics, args.nc, args.device)
    args.img_size = shape[1]
    args.loaders = (train_loader, test_loader)
    args.shape = shape

    train(args)
