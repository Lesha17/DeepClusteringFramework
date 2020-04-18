import torch

from clustering_tool.modules.clusterers.clusterer import Clusterer


@Clusterer.register("xie_clusterer")
class XieClusterer(Clusterer):
    def __init__(self, num_clusters, embedding_size, cluster_centers = None, alpha=1, trainable_centers=False):
        super(XieClusterer, self).__init__()

        self.alpha = alpha
        self.num_clusters = num_clusters
        if cluster_centers is None:
            cluster_centers = torch.rand((num_clusters, embedding_size))
            cluster_centers /= torch.max(torch.abs(cluster_centers))
        self.cluster_centers = torch.nn.Parameter(cluster_centers, requires_grad=trainable_centers)

    def forward(self, x, h):
        s = self.calculate_s(h)
        output_dict = {'s': s, 'loss': self.calculate_loss(s)}
        return output_dict

    def calculate_s(self, h):
        # shape: (batch_size, num_clusters)
        distances = torch.cdist(h, self.cluster_centers)
        modified_distances = torch.pow(torch.pow(distances, 2)/ self.alpha + 1, -(self.alpha + 1)/2).squeeze()
        denominator = torch.sum(modified_distances, -1).squeeze().unsqueeze(-1).expand(modified_distances.shape)

        #shape: (batch_size, num_clusters)
        s = modified_distances / denominator
        return s

    def calculate_loss(self, s):
        # dimension: (num_clusters)
        f = torch.sum(s, 0)
        # dimension: (batch_size, num_clusters)
        q_numenator = torch.div(torch.pow(s, 2), f)
        # dimension: (batch_size)
        q_denominator = torch.sum(q_numenator, 1).unsqueeze(-1).expand(q_numenator.shape)
        # dimension: (batch_size, num_clusters)
        q = torch.div(q_numenator, q_denominator)
        loss = -torch.nn.functional.kl_div(s, q)
        return loss
