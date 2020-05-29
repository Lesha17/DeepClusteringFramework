import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from clustering_tool.metrics import calculate_metrics


def constant_loss_weight_fn(weight):
    return lambda epoch : weight

def linear_weight_fn(initial_weight, step):
    return lambda epoch : initial_weight + epoch * step

def exponential_weight_fn(initial_weight, gamma):
    return lambda epoch : initial_weight * (gamma ** epoch)

def plot_losses(losses_dict, stack=True):
    keys, losses = zip(*losses_dict.items())
    plt.figure()
    if stack:
        plt.stackplot(range(len(losses[0])), *losses, labels=keys)
    else:
        for key, loss in losses_dict.items():
            plt.plot(loss, label=key)
    plt.legend()
    plt.show()


def get_batches(model, dataloader):
    x_batches = []
    h_batches = []
    label_batches = []
    for batch in dataloader:
        x = batch['input']
        x_batches.append(x)
        label_batches.append(batch['label'])
        with torch.no_grad():
            h = model(x)['h']
            h_batches.append(h)

    return x_batches, h_batches, label_batches


def train(model, dataloader, losses_weights_fns, lr, gamma=1.0, num_epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr)
    sheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    loss_hist = defaultdict(list)
    metrics_hist = defaultdict(list)
    weighted_loss_hist = defaultdict(list)

    for epoch in tqdm.tqdm(range(num_epochs), disable=True):
        x_batches, h_batches, label_batches = get_batches(model, dataloader)
        for x, h_prev in zip(x_batches, h_batches):
            opt.zero_grad()

            output_dict = model(x, h_prev=h_prev)
            losses_keys = [loss_key for loss_key in output_dict.keys() if loss_key.endswith('loss')]
            losses = {}
            weighted_losses = {}
            for loss_key in losses_keys:
                loss = output_dict[loss_key]
                losses[loss_key] = loss

                loss_weight = losses_weights_fns[loss_key](epoch)
                weighted_losses[loss_key] = loss_weight * loss

            loss = sum(weighted_losses.values())

            loss.backward()
            opt.step()

            for key, l in losses.items():
                loss_hist[key].append(l.item())

            for key, l in weighted_losses.items():
                weighted_loss_hist[key].append(l.item())

        metrics = calculate_metrics(model, dataloader)
        for name, metric in metrics.items():
            metrics_hist[name].append(metric)
        sheduler.step()
