from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import tqdm
import torch
from clustering_tool.clusterer import ClustererWithCenters
import matplotlib.pyplot as plt


def visualize(model, dataloader, figsize=(10, 10)):
    embeds = []
    clusters = []
    for batch in tqdm.tqdm(dataloader, disable=True):
        with torch.no_grad():
            model_output = model(batch['input'])

        batch_embeds = model_output['h']
        batch_clusters = torch.argmax(model_output['s'], dim=1)
        embeds += [emb.numpy() for emb in batch_embeds.cpu()]
        clusters += [cl.item() for cl in batch_clusters]
    embeddings = np.stack(embeds, axis=0)

    cluster_centers = []
    if isinstance(model.clusterer, ClustererWithCenters):
        cluster_centers = model.clusterer.cluster_centers.cpu().detach().numpy()

    if embeddings.shape[1] > 2:
        print('Transforming with PCA')
        pca = PCA(n_components=2, whiten=True)
        emb_n_centers_2d = pca.fit_transform(np.concatenate((embeddings, cluster_centers), axis=0))
        emb2d = emb_n_centers_2d[:-len(cluster_centers)]
        centers2d = emb_n_centers_2d[-len(cluster_centers):]

        plt.figure(figsize=figsize)
        x_min = np.quantile(emb2d[:, 0], 0.01)
        x_max = np.quantile(emb2d[:, 0], 0.99)
        x_margin = (x_max - x_min) * 0.12
        x_min -= x_margin
        x_max += x_margin
        y_min = np.quantile(emb2d[:, 1], 0.01)
        y_max = np.quantile(emb2d[:, 1], 0.99)
        y_margin = (y_max - y_min) * 0.12
        y_min -= y_margin
        y_max += y_margin
        plt.xlim(left=x_min, right=x_max)
        plt.ylim(bottom=y_min, top=y_max)
        scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=clusters, s=10)
        if len(cluster_centers) > 0:
            plt.scatter(centers2d[:, 0], centers2d[:, 1], marker='s', s=200, c='black')
            scatter = plt.scatter(centers2d[:, 0], centers2d[:, 1], marker='*', s=200,
                                  c=np.arange(0, len(cluster_centers), dtype=int))
        plt.legend(*scatter.legend_elements())
        plt.show()

        print('Transforming with TSNE')
        tsne = TSNE(2)
        emb_n_centers_2d = tsne.fit_transform(np.concatenate((embeddings, cluster_centers), axis=0))
        emb2d = emb_n_centers_2d[:-len(cluster_centers)]
        centers2d = emb_n_centers_2d[-len(cluster_centers):]

        plt.figure(figsize=figsize)
        x_min = np.quantile(emb2d[:, 0], 0.01)
        x_max = np.quantile(emb2d[:, 0], 0.99)
        x_margin = (x_max - x_min) * 0.12
        x_min -= x_margin
        x_max += x_margin
        y_min = np.quantile(emb2d[:, 1], 0.01)
        y_max = np.quantile(emb2d[:, 1], 0.99)
        y_margin = (y_max - y_min) * 0.12
        y_min -= y_margin
        y_max += y_margin
        plt.xlim(left=x_min, right=x_max)
        plt.ylim(bottom=y_min, top=y_max)
        scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=clusters, s=10)
        if len(cluster_centers) > 0:
            plt.scatter(centers2d[:, 0], centers2d[:, 1], marker='s', s=200, c='black')
            scatter = plt.scatter(centers2d[:, 0], centers2d[:, 1], marker='*', s=200,
                                  c=np.arange(0, len(cluster_centers), dtype=int))
        plt.legend(*scatter.legend_elements())
        plt.show()
    else:
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, s=10)
        plt.legend(*scatter.legend_elements())
        if len(cluster_centers) > 0:
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='s', s=200, c='black')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=200,
                        c=np.arange(0, len(cluster_centers), dtype=int))
        plt.show()