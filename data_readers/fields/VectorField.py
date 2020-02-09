import torch
from typing import Dict, List

from allennlp.data import Field
from allennlp.nn import util
from overrides import overrides


class VectorField(Field[Dict[str, torch.Tensor]]):
    def __init__(self, vector, namespace = "vector"):
        super(VectorField, self).__init__()
        self._vector = vector
        self._namespace = namespace

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        return { self._namespace: torch.tensor(self._vector, dtype=torch.float, requires_grad=False) }

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)
