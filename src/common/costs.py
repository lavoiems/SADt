import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter


def cosine(emb1, emb2, idx, *args):
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True)
    return emb1[idx].mm(emb2.transpose(0,1))


def get_avg_dist(emb, query, n=10):
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(n, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1))
    all_distances = torch.cat(all_distances)
    return all_distances


def csls_knn(emb1, emb2, idx, avg_dist1=None, avg_dist2=None):
    if avg_dist1 is None or avg_dist2 is None:
        avg_dist1 = get_avg_dist(emb2, emb1)
        avg_dist2 = get_avg_dist(emb1, emb2)
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True)
    scores = emb1[idx].mm(emb2.t())
    return 2*scores - avg_dist1 - avg_dist2


def gan(discriminator, batch_size, real, device, *data):
    dis = discriminator(*data)
    label = torch.full((batch_size,), real, device=device)
    return F.binary_cross_entropy(dis, label)


def get_top_pairs(emb1, emb2, dist, best_rank, device):
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True)
    best_scores, best_targets = [], []
    avg_dist1 = get_avg_dist(emb2, emb1)
    avg_dist2 = get_avg_dist(emb1, emb2)
    for idx in range(0, len(emb1), 1024):
        scores = dist(emb1, emb2, range(idx, min(len(emb1), idx+1024)), avg_dist1, avg_dist2)
        scores, targets = scores.topk(2, dim=1, largest=True, sorted=True)
        best_scores.append(scores)
        best_targets.append(targets)
    best_scores = torch.cat(best_scores, 0)
    best_targets = torch.cat(best_targets, 0)
    pairs = torch.cat((
        torch.arange(0, best_targets.size(0), device=device).long().unsqueeze(1),
        best_targets[:, 0].unsqueeze(1)
    ), 1)
    diff = best_scores[:, 0] - best_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    best_scores, pairs = best_scores[reordered], pairs[reordered]
    selected = pairs.max(1)[0] <= best_rank
    mask = selected.unsqueeze(1).expand_as(best_scores)
    return pairs.masked_select(mask).view(-1, 2)


def dist_mean_cosine(src_emb, tgt_emb, mapping, dist, device, best_rank):
    """
    Mean-cosine model selection criterion.
    """
    # get normalized embeddings
    src_emb = mapping(src_emb).data
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    # build dictionary
    dico_max_size = 10000
    # temp params / dictionary generation
    dico = get_top_pairs(src_emb, tgt_emb, dist, best_rank, device)
    mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
    mean_cosine = mean_cosine.item()
    return mean_cosine


def unsupervised_mean_cosine(emb1, emb2, transfer, dist, device, best_rank=10000):
    return dist_mean_cosine(emb1, emb2, transfer, dist, device, best_rank)


def load_dictionary(pairs, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    # sort the dictionary by source word frequencies
    pairs = sorted(zip(*pairs), key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(word2id1, emb1, dictionary, word2id2, emb2, method):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    dico = load_dictionary(dictionary, word2id1, word2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # contextual dissimilarity measure
    elif method == 'csls':
        # average distances to k nearest neighbors
        knn = 10
        average_dist1 = get_avg_dist(emb2, emb1, knn)
        average_dist2 = get_avg_dist(emb1, emb2, knn)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(1, 1, True)[1]
    for k in [1]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        results.append(('precision_at_%i' % k, precision_at_k))
    return results


def word_accuracy(k, emb1, emb2, translation, word2id1, word2id2, dictionary, dist, device):
    return get_word_translation_accuracy(word2id1, translation(emb1), dictionary, word2id2, emb2, dist)[0][1]


# def word_accuracy(k, emb1, emb2, translation, word2id1, word2id2, dictionary, dist, device):
#     src_idx = torch.LongTensor(itemgetter(*dictionary[0])(word2id1))
#     scores = dist(translation(emb1), emb2, src_idx)
#     top_scores = scores.topk(k, 1)[1].to('cpu').numpy()
#     tgt_idx = np.repeat(np.array(itemgetter(*dictionary[1])(word2id2))[:, None], top_scores.shape[1], 1)
#     all_correct = (tgt_idx == top_scores).sum(1)
#     correct = {}
#     for j, idx in enumerate(src_idx.cpu().tolist()):
#         correct[idx] = min(correct.get(idx, 0) + all_correct[j], 1)
#     return np.mean(list(correct.values()))
