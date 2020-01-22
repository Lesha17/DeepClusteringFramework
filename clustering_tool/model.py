from allennlp.models.model import Model
from allennlp.modules import Embedding
from overrides import overrides
from torch.nn.modules import Module

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2seq_decoders import SeqDecoder
from allennlp.models.encoder_decoders.composed_seq2seq import ComposedSeq2Seq


class Clusterer(Module):
    """
    Defines a cluster assignments calculation method from given encoder input (x) and output (h)
    """

    def __init__(self):
        super(Clusterer, self).__init__()

    def forward(self, x, h):
        """
        Calculates cluster assignments from given encoder input and output
        TODO specify arguments and outputs shapes

        :param x: given encoder input
        :param h: given encoder output
        :return:
        -------
        Dict with keys:
        ``'s'``: ``List[torch.Tensor]``
            A cluster assignments for given samples, required
        """
        raise NotImplementedError


class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def loss(self, x, h, decoded_x, s):
        raise NotImplementedError


@Model.register("deep_clustering")
class DeepClusteringModel(Model):
    """
    Performs deep embedding clustering using given loss
    """

    def __init__(self, vocab: Vocabulary,
                 encoder : Seq2VecEncoder,
                 decoder : SeqDecoder,
                 clusterer: Clusterer,
                 loss: Loss,
                 embedder: Embedding = None):
        """

        :param vocab:
        :param encoder: an encoder f(x): X -> H of input x; Maps high dimension x into low dimension h to cluster
        :param decoder: a decoder g(h) H -> X; maps encoder output back to sample's initial high - dimension space
        :param clusterer: computes soft cluster assignments s for given h
        :param loss: a joint autoencoder and clusterer loss; TODO maybe separate clusterer and autoencoder loss? Move clustering loss into clusterer
        :param embedder: an input sequence token embedder. Must not be trainable
        """

        super().__init__(vocab)

        self._encoder = encoder
        self._decoder = decoder
        self._clusterer = clusterer
        self._loss = loss
        self._embedder = embedder

    @overrides
    def forward(self, input_sequence):
        if self._embedder:
            x = self._embedder(input_sequence)
        else:
            x = input_sequence

        # shape: (batch_size, bottleneck_embedding_size)
        h = self._encoder(x)
        # shape: (batch_size, clusters_num)
        s = self._clusterer(h)

        output_dict = {}
        if self.training:
            # shape: (batch_size, max_seq_length, input_embedding_size)
            decoded_x = self._decoder(h)
            output_dict['loss'] = self._loss(x, h, decoded_x, s)

        output_dict['predictions'] = s

        return output_dict
