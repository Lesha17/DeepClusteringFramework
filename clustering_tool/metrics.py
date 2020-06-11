import os
import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    logreg.fit(h_train, labels_train)

    probas_predict = logreg.predict_proba(h_test)
    labels_predict = logreg.predict(h_test)

    return {'acc': accuracy_score(labels_test, labels_predict),
            'f1': f1_score(labels_test, labels_predict, average='weighted'),
            'roc_auc': roc_auc_score(labels_test, probas_predict, average='weighted', multi_class='ovr')}


def _parse_file(metrics_filename, data_name, emb_name, attempt, loss_name):
    if not os.path.exists(metrics_filename):
        print('No file:', metrics_filename)
        return None

    with open(metrics_filename, 'r') as file:
        metrics = eval(file.readline())
        line = file.readline()
        row = {'data': data_name, 'embed': emb_name, 'attempt': attempt, 'loss': loss_name}
        row.update(metrics)
        if line:
            classification = eval(line + '\n')
            row['f1'] = classification['f1']
            row['roc-auc'] = classification['roc_auc']

    return row

def aggregate_metrics(ouput_file, metrics_dir = 'output/metrics', num_attempts=5):
    results = []

    for data_name in 'SearchSnippets,Biomedical,StackOverflow'.split(','):
        for emb_name in 'bert_cls,bert_avg,bert_sif,bert_max'.split(','):
            for attempt in range(num_attempts):
                metrics_filename = os.path.join(metrics_dir, f'metrics_{data_name}_{emb_name}_{attempt}_initial.json')
                row = _parse_file(metrics_filename, data_name, emb_name, attempt, 'initial')
                if row is not None:
                    row['decoder'] = False
                    results.append(row)

                for loss_name in 'kl_div,cross_entropy,bce,dot_product'.split(','):
                    metrics_filename = os.path.join(metrics_dir,
                                                    f'metrics_{data_name}_{emb_name}_{attempt}_{loss_name}.json')

                    row = _parse_file(metrics_filename, data_name, emb_name, attempt, loss_name)
                    if row is not None:
                        row['decoder'] = False
                        results.append(row)

                    metrics_filename = os.path.join(metrics_dir,
                                                    f'metrics_{data_name}_{emb_name}_{attempt}_{loss_name}_dec.json')
                    row = _parse_file(metrics_filename, data_name, emb_name, attempt, loss_name)
                    if row is not None:
                        row['decoder'] = True
                        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(ouput_file)

def metrics_to_tex(metrics_file, outfile):
    prev_data = ''
    prev_embed = ''
    prev_loss = ''

    LOSSES = {'cross_entropy': 'CE', 'bce': 'BCE', 'kl_div': 'KL', 'dot_product': '$\cdot$', 'initial': '-'}

    df = pd.read_csv(metrics_file, index_col='Unnamed: 0')
    df_grouped = df.groupby(['data', 'embed', 'loss', 'decoder'], sort=False).agg(['mean', 'std']).drop(['attempt'], axis=1) * 100

    with open(outfile, 'w') as f:
        for index, row in df_grouped.iterrows():
            data, embed, loss, decoder = index
            embed_str = embed.replace('bert_', '') if prev_embed != embed else ''
            loss_str = LOSSES[loss] if prev_loss != loss else ''
            decoder_str = '+' if decoder else '-'

            str = f"{embed_str} & {loss_str} & {decoder_str}" + \
                  f"& ${row['ACC']['mean']:.1f} \\pm {row['ACC']['std']:.1f}$ & ${row['NMI']['mean']:.1f} \\pm {row['NMI']['std']:.1f}$" + \
                  f"& ${row['f1']['mean']:.1f} \\pm {row['f1']['std']:.1f}$ & ${row['roc-auc']['mean']:.1f} \\pm {row['roc-auc']['std']:.1f}$"

            if prev_data != data:
                f.write(data + '\n')

            f.write(str)
            f.write('\\\\ \n')

            prev_data = data
            prev_embed = embed
            prev_loss = loss
