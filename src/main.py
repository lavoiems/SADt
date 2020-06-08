import time
import argparse
import torch
import numpy as np
from common.util import set_paths, create_paths, dump_args
from models import classifier
from models import gan
from models import vmt
from models import vrinv
from models import vmt_cluster
from models import udt
from models import vmtc_repr
from models import i2i

_models_ = {
    'classifier': classifier,
    'gan': gan,
    'vmt': vmt,
    'vrinv': vrinv,
    'vmt_cluster': vmt_cluster,
    'udt': udt,
    'vmtc_repr': vmtc_repr,
    'i2i': i2i,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--run-id', type=str, default=str(time.time()))
    parser.add_argument('--root-path', default='./experiments/')
    parser.add_argument('--server', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--visdom_dir', type=str, default='.')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=50001)
    parser.add_argument('--evaluate', type=int, default=100)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--valid-split', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    for name, model in _models_.items():
        model_parser = subparsers.add_parser(name)
        model.parse_args(model_parser)
        model_parser.set_defaults(func=model.execute)
        model_parser.set_defaults(model=name)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    args.run_name = f'{args.exp_name}_{args.run_id}-{args.seed}'
    set_paths(args)
    create_paths(args.save_path, args.model_path, args.log_path)
    dump_args(args)

    if args.visdom:
        from common.visualise import Visualiser
        args.visualiser = Visualiser(args.server, args.port, f'{args.exp_name}_{args.run_id}', args.reload, '.')
    else:
        from common.tensorboard import Visualiser
        args.visualiser = Visualiser(args.log_path, args.exp_name)
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    args.visualiser.args(args)
    args.func(args)
