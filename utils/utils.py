from collections import defaultdict


def map_score(qids, labels, preds):
    """"""
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.items():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


def mean_reciprocal_rank(qids, labels, preds):
    """"""
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    mrr = 0.
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                mrr += 1. / i
                break

    mrr /= len(qid2cand)
    return mrr


def precision_at_k(qids, labels, preds, k=1):
    """"""
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    p_at_k = 0.0
    good_qids = []
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if i == k and label > 0:
                p_at_k += 1.
                good_qids.append(qid)
            if i > k:
                break

    p_at_k /= len(qid2cand)
    return p_at_k
