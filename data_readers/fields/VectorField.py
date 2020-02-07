import torch
from typing import Dict

from allennlp.data import Field
from overrides import overrides


class VectorField(Field):
    def __init__(self, vector):
        super(VectorField, self).__init__()
        self._vector = vector

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]):
        return torch.tensor(self._vector, dtype=torch.float, requires_grad=False)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}
