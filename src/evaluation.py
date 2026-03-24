import pandas as pd
import numpy as np

def evaluate_preclustered(
    df: pd.DataFrame,
    true_cluster_col: str = "cluster_true",
    pred_cluster_col: str = "cluster",
    compute_link_metrics: bool = False
):
    d = df[[true_cluster_col, pred_cluster_col]].copy()

    def _to_int_codes(series):
        return pd.Categorical(series).codes

    y_true = _to_int_codes(d[true_cluster_col])
    y_pred = _to_int_codes(d[pred_cluster_col])

    def contingency(y_true, y_pred):
        a = pd.Series(y_true)
        b = pd.Series(y_pred)
        C = pd.crosstab(a, b).to_numpy(dtype=np.int64)
        return C

    C = contingency(y_true, y_pred)
    ni = C.sum(axis=1)
    nj = C.sum(axis=0)
    N = C.sum()

    def _entropy(counts):
        if counts.sum() == 0:
            return 0.0
        p = counts[counts > 0] / counts.sum()
        return float(-np.sum(p * np.log(p)))

    def _mutual_info(C):
        N = C.sum()
        if N == 0:
            return 0.0
        pi = C.sum(axis=1) / N
        pj = C.sum(axis=0) / N
        P = C / N
        outer = np.outer(pi, pj)
        mask = (P > 0) & (outer > 0)
        return float(np.sum(P[mask] * np.log(P[mask] / outer[mask])))

    Hx = _entropy(ni)
    Hy = _entropy(nj)
    Ixy = _mutual_info(C)

    with np.errstate(divide="ignore", invalid="ignore"):
        homogeneity = Ixy / Hx if Hx > 0 else 1.0
        completeness = Ixy / Hy if Hy > 0 else 1.0

        if homogeneity + completeness == 0:
            v_measure = 0.0
        else:
            v_measure = (2 * homogeneity * completeness) / (homogeneity + completeness)

    VI = Hx + Hy - 2 * Ixy
    one_minus_VI = 1.0 - VI
    NMI = (2 * Ixy) / (Hx + Hy) if (Hx + Hy) > 0 else 0.0

    def _nC2(x):
        x = x.astype(np.int64)
        return (x * (x - 1)) // 2

    sum_nijC2 = _nC2(C).sum()
    sum_niC2 = _nC2(ni).sum()
    sum_njC2 = _nC2(nj).sum()
    totalC2 = _nC2(np.array([N]))[0]

    expected = (sum_niC2 * sum_njC2) / totalC2 if totalC2 > 0 else 0.0
    max_index = 0.5 * (sum_niC2 + sum_njC2)
    ARI = (sum_nijC2 - expected) / (max_index - expected) if (max_index - expected) != 0 else 0.0

    def one_to_one(C):
        if C.size == 0:
            return 0.0
        try:
            from scipy.optimize import linear_sum_assignment
            max_w = C.max()
            cost = (max_w - C)
            r, c = linear_sum_assignment(cost)
            matched = C[r, c].sum()
        except Exception:
            C_ = C.copy()
            matched = 0
            used_i, used_j = set(), set()
            while True:
                C_[list(used_i), :] = -1
                C_[:, list(used_j)] = -1
                i, j = np.unravel_index(np.argmax(C_, axis=None), C_.shape)
                if C_[i, j] <= 0:
                    break
                matched += C_[i, j]
                used_i.add(i); used_j.add(j)
        return float(matched) / N if N > 0 else 0.0

    one_one = one_to_one(C)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (ni[:, None] + nj[None, :]).astype(float)
        F1_mat = np.where(denom > 0, 2.0 * C / denom, 0.0)
    
    best_per_true = F1_mat.max(axis=1) if F1_mat.size else np.array([0.0])
    shen_F = float(np.sum((ni / N) * best_per_true)) if N > 0 else 0.0

    def _clusters_as_sets(labels):
        lab = pd.Categorical(labels).codes
        groups = {}
        for i, lab_i in enumerate(lab):
            groups.setdefault(lab_i, []).append(i)
        return {frozenset(v) for v in groups.values()}

    gold_sets = _clusters_as_sets(y_true)
    pred_sets = _clusters_as_sets(y_pred)
    tp = len(gold_sets & pred_sets)
    prec_c = tp / len(pred_sets) if len(pred_sets) else 0.0
    rec_c = tp / len(gold_sets) if len(gold_sets) else 0.0
    f1_c = 2*prec_c*rec_c / (prec_c + rec_c) if (prec_c + rec_c) else 0.0

    metrics = {
        "ARI": ARI, "NMI": NMI, "VI": VI, "Homogeneity": homogeneity,
        "Completeness": completeness, "V-Measure": v_measure,
        "1-1": one_one, "S-F": shen_F,
        "cluster_exact_P": prec_c, "cluster_exact_R": rec_c, "cluster_exact_F1": f1_c,
    }

    if compute_link_metrics:
        def _pairs_from_labels(labels):
            lab = pd.Categorical(labels).codes
            by = {}
            for i, l in enumerate(lab):
                by.setdefault(l, []).append(i)
            pairs = set()
            for members in by.values():
                if len(members) >= 2:
                    for a in range(len(members)):
                        for b in range(a+1, len(members)):
                            x, y = members[a], members[b]
                            pairs.add((x, y) if x < y else (y, x))
            return pairs

        gold_pairs = _pairs_from_labels(y_true)
        pred_pairs = _pairs_from_labels(y_pred)
        tp_l = len(gold_pairs & pred_pairs)
        prec_l = tp_l / len(pred_pairs) if len(pred_pairs) else 0.0
        rec_l = tp_l / len(gold_pairs) if len(gold_pairs) else 0.0
        f1_l = 2*prec_l*rec_l / (prec_l + rec_l) if (prec_l + rec_l) else 0.0
        metrics.update({"link_P": prec_l, "link_R": rec_l, "link_F1": f1_l})

    return metrics
