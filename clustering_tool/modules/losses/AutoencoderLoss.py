from allennlp.common import Registrable
from torch.nn.modules.loss import _Loss


class AbstractLoss(Registrable):
    def __init__(self):
        super(AbstractLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def loss(self, *args, **kwargs):
        raise NotImplementedError

class AutoencoderLoss(AbstractLoss):
    def loss(self, x, h, decoded_x):
        raise NotImplementedError


@AutoencoderLoss.register("torch_loss_wrapper")
class TorchLossWrapper(AutoencoderLoss):
    def __init__(self, torch_loss: _Loss):
        super(TorchLossWrapper, self).__init__()
        self._loss = torch_loss

    def loss(self, x, h, decoded_x):
        return self._loss(x, decoded_x)
