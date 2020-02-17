# Model, hyperparameters and training procedure taken from: https://github.com/yunjey/mnist-svhn-transfer
import time
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models, one_hot_embedding
from common.initialize import initialize, infer_iteration
from . import model


def disc_loss(data1, data2, generator, discriminator):
    gen = generator(data1).detach()
    pos_dis = discriminator(data2)
    neg_dis = discriminator(gen)
    pos_loss = torch.mean((pos_dis-1)**2)
    neg_loss = torch.mean(neg_dis**2)
    return pos_loss + neg_loss


def generator_loss(data1, generator1, generator2, discriminator):
    gen = generator1(data1)
    recon = generator2(gen)
    neg_dis = discriminator(gen)
    neg_loss = torch.mean((neg_dis-1)**2)
    recon_loss = torch.mean((data1 - recon)**2)
    return neg_loss + recon_loss


def define_models(shape1, **parameters):
    g12 = model.G()
    g21 = model.G()
    d1 = model.D()
    d2 = model.D()
    return {
        'g12': g12,
        'g21': g21,
        'd1': d1,
        'd2': d2
    }


@torch.no_grad()
def evaluate_cluster_accuracy(visualiser, i, loader, class_map, classifier, id, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        pred = class_map[pred]
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Clustering accuracy {id}', step=i)
    return accuracy


@torch.no_grad()
def evaluate_gen_class_accuracy(visualiser, i, loader, classifier, generator, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        gen = generator(data)
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Generated accuracy', step=i)
    return accuracy


@torch.no_grad()
def evaluate_class_accuracy(visualiser, i, loader, classifier, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Target test accuracy', step=i)
    return accuracy

@torch.no_grad()
def evaluate_cluster(visualiser, i, nc, loader, classifier, id, device):
    labels = []
    preds = []
    n_preds = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1)
        labels += [label]
        preds += [pred]
        n_preds += len(pred)
    labels = torch.cat(labels)
    preds = torch.cat(preds).argmax(1)
    correct = 0
    total = 0
    cluster_map = []
    for j in range(nc):
        label = labels[preds == j]
        if len(label):
            l = one_hot_embedding(label, nc).sum(0)
            correct += l.max()
            cluster_map.append(l.argmax())
        total += len(label)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Transfer clustering accuracy {id}', step=i)
    return torch.LongTensor(cluster_map).to(device)


@torch.no_grad()
def evaluate(visualiser, data1, generator, id):
    visualiser.image(data1.cpu().numpy(), f'target{id}', 0)
    X = generator(data1)
    visualiser.image(X.cpu().numpy(), f'data{id}', 0)
    merged = len(X)*2 * [None]
    merged[:2*len(data1):2] = data1
    merged[1:2*len(X):2] = X
    merged = torch.stack(merged)
    visualiser.image(merged.cpu().numpy(), f'Comparison{id}', 0)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    g12 = models['g12'].to(args.device)
    g21 = models['g21'].to(args.device)
    d1 = models['d1'].to(args.device)
    d2 = models['d2'].to(args.device)
    eval_model = args.evaluation.eval().to(args.device)
    print(g12)
    print(g21)
    print(d1)
    print(d2)

    optim_g12 = optim.Adam(g12.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_g21 = optim.Adam(g21.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_d1 = optim.Adam(d1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_d2 = optim.Adam(d2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    titer1 = iter(test_loader1)
    titer2 = iter(test_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        g12.train()
        g21.train()
        d1.train()
        d2.train()
        batchx, iter1 = sample(iter1, train_loader1)
        data1 = batchx[0].to(args.device)
        if data1.shape[0] != args.train_batch_size:
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        data2 = batchy[0].to(args.device)
        if data2.shape[0] != args.train_batch_size:
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)

        dloss1 = disc_loss(data2, data1, g21, d1)
        optim_d1.zero_grad()
        dloss1.backward()
        optim_d1.step()

        dloss2 = disc_loss(data1, data2, g12, d2)
        optim_d2.zero_grad()
        dloss2.backward()
        optim_d2.step()

        gloss1 = generator_loss(data1, g12, g21, d2)
        optim_g12.zero_grad()
        optim_g21.zero_grad()
        gloss1.backward()
        optim_g12.step()
        optim_g21.step()
        gloss2 = generator_loss(data2, g21, g12, d1)
        optim_g12.zero_grad()
        optim_g21.zero_grad()
        gloss2.backward()
        optim_g12.step()
        optim_g21.step()

        if i % args.evaluate == 0:
            g12.eval()
            g21.eval()
            batchx, titer1 = sample(titer1, test_loader1)
            data1 = batchx[0].to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            data2 = batchy[0].to(args.device)
            print('Iter: %s' % i, time.time() - t0)
            evaluate(args.visualiser, data1, g12, 'x')
            evaluate(args.visualiser, data2, g21, 'y')
            evaluate_gen_class_accuracy(args.visualiser, i, test_loader1, eval_model, g12, args.device)
            evaluate_class_accuracy(args.visualiser, i, test_loader2, eval_model, args.device)
            args.visualiser.plot(dloss1.cpu().detach().numpy(), title=f'Discriminator loss1', step=i)
            args.visualiser.plot(dloss2.cpu().detach().numpy(), title=f'Discriminator loss2', step=i)
            args.visualiser.plot(gloss1.cpu().detach().numpy(), title=f'Generator loss 1-2', step=i)
            args.visualiser.plot(gloss2.cpu().detach().numpy(), title=f'Generator loss 2-1', step=i)
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
