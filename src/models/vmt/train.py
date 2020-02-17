import os
import time
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def soft_cross_entropy(preds, soft_targets):
    return torch.sum(-F.softmax(soft_targets, 1)*F.log_softmax(preds, 1), 1)


def classification_loss(data, classes, classifier):
    pred = classifier(data)
    return F.cross_entropy(pred, classes)


def classification_target_loss(data, classifier):
    preds = classifier(data)
    return soft_cross_entropy(preds, preds).mean()


def disc_loss(data1, data2, discriminator, embedding, classifier, device):
    emb1 = embedding(data1).detach()
    emb2 = embedding(data2).detach()
    c1 = classifier(emb1).detach()
    c2 = classifier(emb2).detach()
    pos_dis = discriminator(emb1, c1)
    neg_dis = discriminator(emb2, c2)
    ones = torch.ones_like(pos_dis, device=device)
    zeros = torch.zeros_like(neg_dis, device=device)
    pos_loss = F.binary_cross_entropy_with_logits(pos_dis, ones)
    neg_loss = F.binary_cross_entropy_with_logits(neg_dis, zeros)
    return 0.5*pos_loss + 0.5*neg_loss


def embed_div_loss(data1, data2, discriminator, embedding, classifier, device):
    emb1 = embedding(data1)
    emb2 = embedding(data2)
    c1 = classifier(emb1)
    c2 = classifier(emb2)
    pos_dis = discriminator(emb1, c1)
    neg_dis = discriminator(emb2, c2)
    zeros = torch.zeros_like(pos_dis, device=device)
    ones = torch.ones_like(neg_dis, device=device)
    pos_loss = F.binary_cross_entropy_with_logits(pos_dis, zeros)
    neg_loss = F.binary_cross_entropy_with_logits(neg_dis, ones)
    return 0.5*pos_loss + 0.5*neg_loss


def compute_perturb(x, y, radius, classifier, device):
    eps = 1e-6 * F.normalize(torch.randn_like(x, device=device))
    eps.requires_grad=True
    xe = x + eps
    ye = classifier(xe)
    loss = soft_cross_entropy(ye, y)
    grad = torch.autograd.grad(loss, eps, grad_outputs=torch.ones_like(loss, device=device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = F.normalize(grad)
    x_prime = radius*grad + x
    return x_prime.detach()


def vat_loss(x, classifier, radius, device):
    y = classifier(x)
    x_prime = compute_perturb(x, y, radius, classifier, device)
    y_prime = classifier(x_prime)
    return soft_cross_entropy(y_prime, y.detach()).mean()


def mixup_loss(x, classifier, device):
    alpha = torch.rand(x.shape[0], device=device)
    alphax = alpha.view(-1, 1, 1, 1)
    alphay = alpha.view(-1, 1)
    idx = torch.randperm(len(x), device=device)
    x2 = x[idx]
    y = classifier(x)
    y2 = y[idx]

    mix_x = alphax*x + (1-alphax)*x2
    mix_y = alphay*y + (1-alphay)*y2

    mix_yp = classifier(mix_x)
    return soft_cross_entropy(mix_yp, mix_y.detach()).mean()


def define_models(shape1, **parameters):
    discriminator = model.Discriminator(shape1[0], shape1[1], **parameters)
    classifier = model.Classifier(shape1[0], **parameters)
    return {
        'classifier': classifier,
        'discriminator': discriminator,
    }


@torch.no_grad()
def evaluate(loader, classifier, device):
    correct = 0
    total = 0
    for data, label in loader:
        if data.shape[0] < 2:
            continue
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    return accuracy


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    classifier = models['classifier'].to(args.device)
    discriminator = models['discriminator'].to(args.device)
    print(classifier)
    print(discriminator)

    optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_classifier = optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optims = {'optim_discriminator': optim_discriminator, 'optim_classifier': optim_classifier}
    initialize(optims, args.reload, args.save_path, args.model_path)

    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        classifier.train()
        discriminator.train()
        batchx, iter1 = sample(iter1, train_loader1)
        data1 = batchx[0].to(args.device)
        if data1.shape[0] != args.train_batch_size:
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)
        label = batchx[1].to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        data2 = batchy[0].to(args.device)
        if data2.shape[0] != args.train_batch_size:
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)

        optim_discriminator.zero_grad()
        d_loss = disc_loss(data1, data2, discriminator, classifier.x, classifier.mlp, args.device)
        d_loss.backward()
        optim_discriminator.step()

        optim_classifier.zero_grad()
        c_loss = classification_loss(data1, label, classifier)
        tcw_loss = classification_target_loss(data2, classifier)
        dw_loss = embed_div_loss(data1, data2, discriminator, classifier.x, classifier.mlp, args.device)
        v_loss1 = vat_loss(data1, classifier, args.radius, args.device)
        v_loss2 = vat_loss(data2, classifier, args.radius, args.device)
        m_loss1 = mixup_loss(data1, classifier, args.device)
        m_loss2 = mixup_loss(data2, classifier, args.device)
        (args.cw *c_loss)  .backward()
        (args.tcw*tcw_loss).backward()
        (args.dw *dw_loss) .backward()
        (args.svw*v_loss1) .backward()
        (args.tvw*v_loss2) .backward()
        (args.smw*m_loss1) .backward()
        (args.tmw*m_loss2) .backward()
        optim_classifier.step()

        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0)
            classifier.eval()

            test_accuracy_x = evaluate(test_loader1, classifier, args.device)
            test_accuracy_y = evaluate(test_loader2, classifier, args.device)

            save_path = args.save_path
            with open(os.path.join(save_path, 'c_loss'), 'a') as f: f.write(f'{i},{c_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'tcw_loss'), 'a') as f: f.write(f'{i},{tcw_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'dw_loss'), 'a') as f: f.write(f'{i},{dw_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'v_loss1'), 'a') as f: f.write(f'{i},{v_loss1.cpu().item()}\n')
            with open(os.path.join(save_path, 'v_loss2'), 'a') as f: f.write(f'{i},{v_loss2.cpu().item()}\n')
            with open(os.path.join(save_path, 'm_loss1'), 'a') as f: f.write(f'{i},{m_loss1.cpu().item()}\n')
            with open(os.path.join(save_path, 'm_loss2'), 'a') as f: f.write(f'{i},{m_loss2.cpu().item()}\n')
            with open(os.path.join(save_path, 'd_loss2'), 'a') as f: f.write(f'{i},{d_loss.cpu().item()}\n')
            #with open(os.path.join(save_path, 'eval_accuracy_x'), 'a') as f: f.write(f'{i},{eval_accuracy_x}\n')
            #with open(os.path.join(save_path, 'eval_accuracy_y'), 'a') as f: f.write(f'{i},{eval_accuracy_y}\n')
            with open(os.path.join(save_path, 'test_accuracy_x'), 'a') as f: f.write(f'{i},{test_accuracy_x}\n')
            with open(os.path.join(save_path, 'test_accuracy_y'), 'a') as f: f.write(f'{i},{test_accuracy_y}\n')
            args.visualiser.plot(c_loss.cpu().detach().numpy(), title='Source classifier loss', step=i)
            args.visualiser.plot(tcw_loss.cpu().detach().numpy(), title='Target classifier cross entropy', step=i)
            args.visualiser.plot(dw_loss.cpu().detach().numpy(), title='Classifier marginal divergence', step=i)
            args.visualiser.plot(v_loss1.cpu().detach().numpy(), title='Source virtual adversarial loss', step=i)
            args.visualiser.plot(v_loss2.cpu().detach().numpy(), title='Target virtual adversarial loss', step=i)
            args.visualiser.plot(m_loss1.cpu().detach().numpy(), title='Source mix up loss', step=i)
            args.visualiser.plot(m_loss2.cpu().detach().numpy(), title='Target mix up loss', step=i)
            args.visualiser.plot(d_loss.cpu().detach().numpy(), title='Discriminator loss', step=i)
            #args.visualiser.plot(eval_accuracy_x, title='Eval acc X', step=i)
            #args.visualiser.plot(eval_accuracy_y, title='Eval acc Y', step=i)
            args.visualiser.plot(test_accuracy_x, title='Test acc X', step=i)
            args.visualiser.plot(test_accuracy_y, title='Test acc Y', step=i)
            t0 = time.time()
            save_models(models, i, args.model_path, args.evaluate)
            save_models(optims, i, args.model_path, args.evaluate)
