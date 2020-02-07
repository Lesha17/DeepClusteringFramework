from allennlp.training.metrics import Metric
import torch


class NormalizedMutualInformation(Metric):
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.reset()

    def __call__(self, x, y, mask = None):
        y = y.squeeze()
        self.joint_distr += torch.matmul(torch.t(x), y)
        self.x += torch.sum(x, 0)
        self.y += torch.sum(y, 0)

    def get_metric(self, reset: bool):
        x = self.x / sum(self.x)
        y = self.y / sum(self.y)

        xy_distr = torch.matmul(x.unsqueeze(-1), y.unsqueeze(0))
        xy_distr = xy_distr.view(self.x_dim * self.y_dim)

        joint = self.joint_distr.view(self.joint_distr.shape[0] * self.joint_distr.shape[1])
        joint /= torch.sum(joint)

        kl = torch.nn.functional.kl_div(joint, xy_distr)
        x[x < 1e-12] = 1e-12
        y[y < 1e-12] = 1e-12

        x_entropy = torch.sum(torch.dot(x, torch.log(x)))
        y_entropy = torch.sum(torch.dot(y, torch.log(y)))
        if reset:
            self.reset()
        return (2 * kl / (x_entropy + y_entropy)).detach().numpy().item()



    def reset(self):
        self.x = torch.zeros(self.x_dim, dtype=torch.float)
        self.y = torch.zeros(self.y_dim, dtype=torch.float)
        self.joint_distr = torch.zeros((self.x_dim, self.y_dim), dtype=torch.float)
