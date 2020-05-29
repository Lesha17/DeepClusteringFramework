import torch
from IPython.display import clear_output
from collections import defaultdict
from clustering_tool.train import plot_losses
import tqdm

def init(m, noize=0.01):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=noize)
        torch.nn.init.normal_(m.bias, std=noize)

class FeedForwardEncoder(torch.nn.Module):
    def __init__(self, encoder_net, regularization_loss = None):
        super(FeedForwardEncoder, self).__init__()
        self.encoder_net = encoder_net
        self.regularization_loss = regularization_loss

    def forward(self, x, h_prev=None):
        h = self.encoder_net(x)

        output_dict = {'h': h}
        if self.regularization_loss is not None and h_prev is not None:
            output_dict['loss'] = self.regularization_loss(h, h_prev)
        return output_dict

def createEncoder(input_size = 768, activation=torch.nn.ELU(), regularization_loss = None):
    encoder_net = torch.nn.Sequential(
        torch.nn.Linear(input_size, 768),
        activation,
        torch.nn.Linear(768, 384),
        activation,
        torch.nn.Linear(384, 192),
        activation,
        torch.nn.Linear(192, 96),
        activation,
        torch.nn.Linear(96, 48),
        activation,
        torch.nn.Linear(48, 24))
    encoder = FeedForwardEncoder(encoder_net, regularization_loss = regularization_loss)
    #encoder.apply(init)
    return encoder

class FeedForwardDecoder(torch.nn.Module):
    def __init__(self, decoder_net, loss_fn):
        super(FeedForwardDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.loss_fn = loss_fn

    def forward(self, h, x):
        decoded_x = self.decoder_net(h)
        loss = self.loss_fn(decoded_x, x)
        return {'decoded_x': decoded_x, 'loss': loss}

def createDecoder(output_size=768, activation = torch.nn.ELU(), loss_fn=torch.nn.MSELoss()):
    decoder_net = torch.nn.Sequential(
        torch.nn.Linear(24, 48),
        activation,
        torch.nn.Linear(48, 96),
        activation,
        torch.nn.Linear(96, 192),
        activation,
        torch.nn.Linear(192, 384),
        activation,
        torch.nn.Linear(384, 768),
        activation,
        torch.nn.Linear(768, output_size),
    )
    decoder = FeedForwardDecoder(decoder_net, loss_fn)
    #decoder.apply(init)
    return decoder



def train_autoencoder(encoder, decoder, dataloader, losses_weights_fns, lr, gamma=1.0, num_epochs=100, decoder_only=False):
    opt = torch.optim.Adam(torch.nn.ModuleList([encoder, decoder]).parameters(), lr)
    sheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    loss_hist = defaultdict(list)

    for epoch in tqdm.tqdm(range(num_epochs), disable=True):
        for batch in dataloader:
            opt.zero_grad()
            x = batch['input']
            encoder_out = encoder(x)
            h = encoder_out['h']
            if decoder_only:
                h = h.detach()
            decoder_out = decoder(h, x)
            x_pred = decoder_out['decoded_x']

            decoder_loss = losses_weights_fns['decoder_loss'](epoch) * decoder_out['loss']
            loss = decoder_loss
            loss_hist['decoder_loss'].append(decoder_loss.item())
            if 'loss' in encoder_out:
                encoder_loss = losses_weights_fns['encoder_loss'](epoch) * encoder_out['loss']
                loss += encoder_loss
                loss_hist['encoder_loss'].append(encoder_loss.item())

            loss.backward()
            opt.step()

        sheduler.step()
        clear_output(wait=True)
