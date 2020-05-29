import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classify(model, dataloader):
    true_labels_arrays = []
    h_arrays = []

    for batch in dataloader:
        with torch.no_grad():
            model_output = model(batch['input'])
        h_arrays.append(model_output['h'].detach().cpu().numpy())

        true_labels_arrays.append(batch['label'].detach().cpu().numpy())

    h = np.concatenate(h_arrays)
    labels = np.concatenate(true_labels_arrays)

    h, labels = shuffle(h, labels)
    h_train, h_test, labels_train, labels_test = train_test_split(h, labels, test_size=0.3)

    logreg = LogisticRegression()
    logreg.fit(h_train, labels_train)

    probas_predict = logreg.predict_proba(h_test)
    labels_predict = logreg.predict(h_test)

    return {'acc': accuracy_score(labels_test, labels_predict),
            'f1': f1_score(labels_test, labels_predict, average='weighted'),
            'roc_auc': roc_auc_score(labels_test, probas_predict, average='weighted', multi_class='ovr')}

