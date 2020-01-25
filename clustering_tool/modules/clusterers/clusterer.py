from allennlp.common import Registrable
from torch.nn import Module


class Clusterer(Module, Registrable):
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
        ``'loss'``
            A clustering loss
        """
        raise NotImplementedError
