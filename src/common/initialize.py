import os
import torch
import json


def initialize(models, reload, dir, model_path):
    for name, model in models.items():
        reload_model = reload and has_models(model_path)
        print(reload_model)
        if reload_model:
            models[name] = load_last_model(model, name, dir)
    return models


def infer_iteration(name, reload, model_path, save_path):
    resume = reload and has_models(model_path)
    if not resume:
        return 0
    names = filter_name(name, save_path)
    epochs = map(parse_model_id, names)
    return max(epochs) + 1


def has_models(path):
    return len(os.listdir(path)) > 0


def normal(model, *args):
    model.apply(normal_init)
    return model


def identity(model, *args):
    model.apply(identity_init)
    return model


def identity_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data[:, :, 1, 1] += 1


def normal_init(m):
    classname = m.__class__.__name__
    if classname.find('Block') != -1:
        return
    if classname.find('Conv2dt ') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('Linear') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_model(shapes, **kwargs):
    model_name = '%s.%s' % (kwargs['model'], 'train')
    model_definition = getattr(__import__(model_name, fromlist='define_models'), 'define_models')
    return model_definition(*shapes, **kwargs)


def create_model(dir, shapes):
    kwargs = json.loads(open(os.path.join(dir, 'args.json'), 'r').read())
    kwargs = sanitize_saved_args(kwargs)
    return define_model(shapes, **kwargs)


def load_last_model(model, model_type, dir):
    names = filter_name(model_type, dir)
    last_name = max(names, key=parse_model_id)
    path = os.path.join(dir, 'model', last_name)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


def sanitize_saved_args(args):
    return dict(map(parameter_arg, args.items()))


def parameter_arg(kv):
    from .utils import get_action
    key, value = kv
    if type(value).__name__ == 'list' and '_parameters' in key:
        return key.split('_parameters')[0], get_action(value)
    return key, value


def create_last_model(dir, shapes, model_type):
    model = create_model(dir, shapes)[model_type]
    model = load_last_model(model, model_type, dir)
    return model


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


def create_all_model(model_dir, shapes, model_type):
    model = create_model(model_dir, shapes)[model_type]
    model.eval()
    checkpoints = filter_name(model_type, model_dir)
    checkpoints = sort_name(checkpoints)
    paths = [os.path.join(model_dir, 'model', f) for f in checkpoints]
    return (load_model(model, path) for path in paths)


def sort_name(names):
    return sorted(names, key=parse_model_id)


def parse_model_id(path):
    return int(path.split('_')[-1])


def filter_name(name, dir):
    model_dir = os.path.join(dir, 'model')
    return filter(lambda x: name in x, os.listdir(model_dir))
