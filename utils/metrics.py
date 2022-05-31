import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster.supervised import check_clusterings
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
from scipy import sparse as sp

nmi = normalized_mutual_info_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


from sklearn.utils.fixes import comb


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def rand_index_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0

    # Compute the RI using the contingency data
    contingency = contingency_matrix(labels_true, labels_pred)

    n = np.sum(np.sum(contingency))
    t1 = comb(n, 2)
    t2 = np.sum(np.sum(np.power(contingency, 2)))
    nis = np.sum(np.power(np.sum(contingency, 0), 2))
    njs = np.sum(np.power(np.sum(contingency, 1), 2))
    t3 = 0.5 * (nis + njs)

    A = t1 + t2 - t3
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    AR = (A - nc) / (t1 - nc)
    return A / t1


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def b3_precision_score(labels_true, labels_pred):
    """Compute the B^3 variant of precision.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    """
    p, _, _ = b3_precision_recall_fscore(labels_true, labels_pred)
    return p


def b3_recall_score(labels_true, labels_pred):
    """Compute the B^3 variant of recall.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float recall: calculated recall
    """
    _, r, _ = b3_precision_recall_fscore(labels_true, labels_pred)
    return r


def f_score(labels_true, labels_pred):
    """Compute the B^3 variant of F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float f_score: calculated F-score
    """
    _, _, f = b3_precision_recall_fscore(labels_true, labels_pred)
    return f


def rand_index(
        labels,
        inst_cluster_id,
        return_counters=False):
    if len(labels) != len(inst_cluster_id):
        if warnings:
            print("Error: length of label and",
                  "inst_clust_id must match.")
        return None

    """
        Rand Index definition:
        let
        f_00 : # of instance pairs with diff classes in diff clusters
        f_01 : # of instance pairs with diff classes in the same cluster
        f_10 : # of instance pairs with same class in diff clusters
        f_11 : # of instance pairs with same classes in the same cluster
        Therefore, the most significative "bit" tells about
        the class label pair and the less significative one
        about the cluster pair.
                Same cluster	Diff Cluster
        Same class	f_11		f_10
        Diff class	f_01		f_00
        then:
        rand_index := (f_00 + f_11) / (f_00 + f_10 + f_01 + f_11)
    """

    # The counter array will express the f_ij value
    # in terms of it's own index. (For example, the
    # counter f_10 is counters[int("10", 2)] == counters[2]
    # and the counter f_11 is counters[int("11", 2)] == counters[3])
    counters = 4 * [0]

    num_inst = len(labels)
    for i in range(num_inst):
        for j in range(num_inst):
            pos = 2 * (labels[i] == labels[j]) + \
                  (inst_cluster_id[i] == inst_cluster_id[j])
            counters[pos] += 1

        # Remove the i == j counter. I do not put an
        # extra "if" to speed up the process.
        counters[3] -= 1  # Remember: int("11", 2) == 3.

    # rand_index = (f_00 + f_11) / (f_00 + f_10 + f_01 + f_11)
    rand_index = (counters[int("00", 2)] + \
                  counters[int("11", 2)]) / sum(counters)

    if return_counters:
        return rand_index, counters

    return rand_index


def randIndex(truth, predicted):

    if len(truth) != len(predicted):
        print("different sizes of the label assignments")
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel - 1):
        for j in range(i + 1, sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same += 1
            count += 1
    return (agree_same + disagree_same) / float(count)
