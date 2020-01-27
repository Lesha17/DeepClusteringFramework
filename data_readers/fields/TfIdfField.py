import torch
from typing import Dict

from allennlp.data import Field
from overrides import overrides
from allennlp.modules.feedforward import FeedForward


class TfIdfField(Field):
    def __init__(self, vector):
        super(TfIdfField, self).__init__()
        self._vector = vector

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]):
        return torch.as_tensor(self._vector)
