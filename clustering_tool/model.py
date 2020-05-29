import logging

import torch

logger = logging.getLogger(__name__)

class DeepClusteringModel(torch.nn.Module):
    """
    Performs deep embedding clustering using given loss
    """

    def __init__(self,
                 encoder,
                 clusterer,
                 decoder = None):
        """

        :param encoder: an encoder f(x): X -> H of input x; Maps high dimension x into low dimension h to cluster
        :param decoder: a decoder g(h) H -> X; maps encoder output back to sample's initial high - dimension space
        :param clusterer: computes soft cluster assignments s for given h
        """

        super(DeepClusteringModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.clusterer = clusterer

        #self.nmi = NormalizedMutualInformation(num_clusters, num_classes if num_classes else num_clusters)

    def forward(self, x, h_prev=None):

        # shape: (batch_size, bottleneck_embedding_size)
        if h_prev is None:
            encoder_out = self.encoder(x)
        else:
            encoder_out = self.encoder(x, h_prev)
        h = encoder_out['h']

        # shape: (batch_size, clusters_num)
        clusterer_out = self.clusterer(x, h)

        output_dict = {'h': h, 's': clusterer_out['s'] }
        if 'loss' in encoder_out:
            output_dict['encoder_loss'] = encoder_out['loss']
        if 'loss' in clusterer_out:
            output_dict['clusterer_loss'] = clusterer_out['loss']

        if self.decoder is not None:
            decoder_out = self.decoder(h, x)
            output_dict['decoded_x'] = decoder_out['decoded_x']
            output_dict['decoder_loss'] = decoder_out.get('loss', None)

        return output_dict



