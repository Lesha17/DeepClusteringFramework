from allennlp.modules import TokenEmbedder
from sklearn.decomposition import PCA
from joblib import load
import torch
import numpy


@TokenEmbedder.register("pca")
class PcaEmbedder(TokenEmbedder):
    def __init__(self, weights_file):
        super(PcaEmbedder, self).__init__()

        pca = load(weights_file)
        if pca.mean_ is not None:
            self.mean = torch.nn.Parameter(torch.tensor(pca.mean_, dtype=torch.float), requires_grad=False)
        else:
            self.mean = None
        self.components = torch.nn.Parameter(torch.tensor(pca.components_.T, dtype=torch.float), requires_grad=False)
        if pca.whiten:
            self.explained_variance = torch.nn.Parameter(torch.tensor(pca.explained_variance_, dtype=torch.float), requires_grad=False)
        else:
            self.explained_variance = None


    def forward(self, data):
        if self.mean is not None:
            data = data - self.mean
        result = torch.matmul(data.unsqueeze(-2), self.components).squeeze()
        if self.explained_variance is not None:
            result = result / torch.sqrt(self.explained_variance)
        return result


    def get_output_dim(self) -> int:
        return self.pca.n_components


