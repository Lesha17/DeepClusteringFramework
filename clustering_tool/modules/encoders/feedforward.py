import torch
from typing import List, Union

from allennlp.common import FromParams
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import feedforward_encoder
from allennlp.nn import Activation
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


@Seq2VecEncoder.register("feedforward")
class MyFeedForward(FeedForward, FromParams):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[Activation, List[Activation]],
                 dropout: Union[float, List[float]] = 0.0) -> None:
        super(MyFeedForward, self).__init__(input_dim, num_layers, hidden_dims, activations, dropout = dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            #print("w: {}, b: {}".format(layer.weight.data, layer.bias.data))
            output = dropout(activation(layer(output)))
        return output
