import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    from sklearn.metrics.pairwise import pairwise_distances

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0

    dist_matrix = pairwise_distances(x)
    n = len(labels)
    sil_values = np.zeros(n)

    for i in range(n):
        cluster_i = labels[i]
        mask_same = labels == cluster_i
        mask_same[i] = False
        count_same = np.sum(mask_same)

        if count_same == 0:
            sil_values[i] = 0.0
            continue

        s_i = np.mean(dist_matrix[i, mask_same])

        other_clusters = unique_labels[unique_labels != cluster_i]
        d_i = np.min([
            np.mean(dist_matrix[i, labels == c])
            for c in other_clusters
        ])

        if s_i == 0 and d_i == 0:
            sil_values[i] = 0.0
        else:
            sil_values[i] = (d_i - s_i) / max(d_i, s_i)

    return np.mean(sil_values)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    n = len(true_labels)
    true_labels = true_labels.reshape(-1)
    predicted_labels = predicted_labels.reshape(-1)

    same_true = true_labels[:, None] == true_labels[None, :]
    same_pred = predicted_labels[:, None] == predicted_labels[None, :]

    correctness = (same_true & same_pred).astype(float)

    same_pred_counts = same_pred.sum(axis=1).astype(float)
    same_pred_counts = np.where(same_pred_counts == 0, 1, same_pred_counts)
    precision = np.where(same_pred.any(axis=1), correctness.sum(axis=1) / same_pred_counts, 0.0)

    same_true_counts = same_true.sum(axis=1).astype(float)
    same_true_counts = np.where(same_true_counts == 0, 1, same_true_counts)
    recall = correctness.sum(axis=1) / same_true_counts

    precision_avg = precision.mean()
    recall_avg = recall.mean()

    if precision_avg + recall_avg == 0:
        return 0.0

    score = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
    return score
