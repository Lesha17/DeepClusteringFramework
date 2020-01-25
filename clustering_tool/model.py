from allennlp.models.model import Model
from allennlp.modules import Embedding
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_decoders import SeqDecoder

from clustering_tool.modules.clusterers.clusterer import Clusterer
from clustering_tool.modules.losses import AutoencoderLoss


@Model.register("deep_clustering")
class DeepClusteringModel(Model):
    """
    Performs deep embedding clustering using given loss
    """

    def __init__(self, vocab: Vocabulary,
                 encoder : Seq2VecEncoder,
                 clusterer: Clusterer,
                 decoder: SeqDecoder = None,
                 autoencoder_loss: AutoencoderLoss = None,
                 embedder: Embedding = None):
        """

        :param vocab:
        :param encoder: an encoder f(x): X -> H of input x; Maps high dimension x into low dimension h to cluster
        :param decoder: a decoder g(h) H -> X; maps encoder output back to sample's initial high - dimension space
        :param clusterer: computes soft cluster assignments s for given h
        :param autoencoder_loss: a joint autoencoder and clusterer loss; TODO maybe separate clusterer and autoencoder loss? Move clustering loss into clusterer
        :param embedder: an input sequence token embedder. Must not be trainable
        """

        super().__init__(vocab)

        self._encoder = encoder
        self._decoder = decoder
        self._clusterer = clusterer
        self._autoencoder_loss = autoencoder_loss
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
        clusterer_out = self._clusterer(h)

        output_dict = {'predictions': clusterer_out['s']}

        if self.training:
            # shape: (batch_size, max_seq_length, input_embedding_size)
            output_dict['loss'] = clusterer_out['loss']
            if self._decoder or self.autoencoder_loss:
                if self._decoder is None or self._autoencoder_loss is None:
                    raise RuntimeError("Decoder and autoencoder loss must be provided together")

                decoded_x = self._decoder(h)
                output_dict["loss"] += self._autoencoder_loss(x, h, decoded_x)

        return output_dict
