import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = zip(*linear_sum_assignment(w.max() - w))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def calculate_metrics(model, dataloader):
    true_labels_arrays = []
    clustered_labels_arrays = []

    for batch in dataloader:
        with torch.no_grad():
            model_output = model(batch['input'])

        true_labels_arrays.append(batch['label'].detach().cpu().numpy())
        clustered_labels_arrays.append(torch.argmax(model_output['s'], dim=1).cpu().numpy())

    true_labels = np.concatenate(true_labels_arrays)
    clustered_labels = np.concatenate(clustered_labels_arrays)

    return {'NMI': normalized_mutual_info_score(true_labels, clustered_labels),
            'ACC': acc(true_labels, clustered_labels)}