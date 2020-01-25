import torch

from clustering_tool.modules.clusterers.clusterer import Clusterer


@Clusterer.register("xie_clusterer")
class XieClusterer(Clusterer):
    def __init__(self, num_clusters, embedding_size, alpha=0.5):
        super(XieClusterer, self).__init__()

        self.alpha = alpha
        self.num_clusters = num_clusters
        self.cluster_centers = torch.nn.Parameter(torch.rand((num_clusters, embedding_size)), requires_grad=True)

        self.reset_parameters()

    def forward(self, x, h):
        output_dict = {'s': self.calculate_s(h)}

        if self.training:
            output_dict['loss'] = self.calculate_loss(h)

        pass

    def calculate_s(self, h):
        # shape: (batch_size, num_clusters)
        distances = torch.cdist(h, self.cluster_centers)
        modified_distances = torch.pow(torch.add(torch.div(torch.pow(distances, 2), self.alpha), 1), -(self.alpha + 1)/2)

        #shape: (batch_size, num_clusters)
        s = torch.div(modified_distances, torch.sum(modified_distances, 1))
        return s

    def calculate_loss(self, h):
        # dimension: (num_clusters)
        f = torch.sum(s, 0)
        # dimension: (batch_size, num_clusters)
        q_numenator = torch.div(torch.pow(s, 2), f)
        # dimension: (batch_size)
        q_denominator = torch.sum(q_numenator, 1)
        # dimension: (batch_size, num_clusters)
        q = torch.div(q_numenator, q_denominator)
        loss = torch.nn.functional.kl_div(q, s, reduction='batchmean')
        return loss
