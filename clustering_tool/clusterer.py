import numpy as np
import torch
import tqdm
from sklearn.cluster import KMeans

def init_cluster_centers(encoder, dataloader, num_clusters, num_iterations=10):
    embeds = []
    for batch in tqdm.tqdm(dataloader, disable=True):
        with torch.no_grad():
            encoder_out = encoder(batch['input'])
        embeds += [emb.numpy() for emb in encoder_out['h'].cpu()]

    loss_sum = 0
    min_loss = np.sum(np.square(np.linalg.norm(embeds, axis=1)))
    best_centers = None
    for i in tqdm.tqdm(range(num_iterations), disable=True):
        kmeans = KMeans(num_clusters)
        kmeans.fit(embeds)
        loss_sum += kmeans.inertia_
        if (kmeans.inertia_ < min_loss
        ):
            min_loss = kmeans.inertia_
            best_centers = kmeans.cluster_centers_

    print('Average:', loss_sum / num_iterations, 'Best: ', min_loss)

    return torch.as_tensor(best_centers, dtype=torch.float)

def kl_div_loss(s, q):
    return torch.nn.functional.kl_div(torch.log(s), q.detach(), reduction='batchmean')

def cross_entropy_loss(s, q):
    return torch.mean(torch.sum(-q.detach() * torch.log(s), dim=1))

def binary_cross_entropy_loss(s, q):
    q = q.detach()
    return torch.mean(torch.sum(-q * torch.log(s) - (1 - q) * torch.log(1 - s), dim=1))

def dot_product_loss(s, q):
    return torch.mean(torch.sum(-q.detach() * s, dim=1))

class ClustererWithCenters():
    def __init__(self):
        self.cluster_centers = None

class XieClusterer(torch.nn.Module, ClustererWithCenters):
    def __init__(self, cluster_centers=None, loss_fn=kl_div_loss, alpha=1, trainable_centers=False):
        super(XieClusterer, self).__init__()

        self.loss_fn = loss_fn
        self.alpha = alpha
        self.cluster_centers = torch.nn.Parameter(cluster_centers, requires_grad=trainable_centers)
        self.loss_func = torch.nn.KLDivLoss()

    def forward(self, x, h):
        s = self.calculate_s(h)
        output_dict = {'s': s, 'loss': self.calculate_loss(s)}
        return output_dict

    def calculate_s(self, h):
        # shape: (batch_size, num_clusters)
        distances = torch.cdist(h, self.cluster_centers)
        # print(f'dist: {distances}')
        modified_distances = torch.pow(torch.pow(distances, 2) / self.alpha + 1, -(self.alpha + 1) / 2).squeeze()
        # print(f'modif dist: {modified_distances}')
        denominator = torch.sum(modified_distances, dim=-1)
        # print(f'denominator: {denominator}')

        # shape: (batch_size, num_clusters)
        s = modified_distances / denominator.unsqueeze(-1)
        return s

    def calculate_loss(self, s):
        # dimension: (num_clusters)
        f = torch.sum(s, dim=0)
        # dimension: (batch_size, num_clusters)
        q_numenator = torch.div(torch.pow(s, 2), f.unsqueeze(0))
        # dimension: (batch_size)
        q_denominator = torch.sum(q_numenator, dim=1)
        # dimension: (batch_size, num_clusters)
        q = torch.div(q_numenator, q_denominator.unsqueeze(-1))
        loss = self.loss_fn(s, q)
        return loss