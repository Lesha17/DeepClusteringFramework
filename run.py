from clustering_tool.autoencoder import *
from clustering_tool.metrics import classify
from clustering_tool.train import *
from clustering_tool.clusterer import *
from clustering_tool.model import *
from clustering_tool.datasets import *
from clustering_tool.embedders.bert import *
from clustering_tool.embedders.common import *
import os

DATEST_TO_NUMCLUSTERS = {'SearchSnippets': 8, 'Biomedical': 20, 'StackOverflow': 20}
LOSSES = {'kl_div': kl_div_loss, 'dot_product': dot_product_loss, 'cross_entropy': cross_entropy_loss, 'bce': binary_cross_entropy_loss}

def create_bert_cls_embedder(bertTokenizer, bertModel, filepath):
    return BertEmbedder(bertTokenizer, bertModel, device=device, embedding_strategy=bert_cls_embeddings)

def create_bert_avg_embedder(bertTokenizer, bertModel, filepath):
    return BertEmbedder(bertTokenizer, bertModel, device=device, embedding_strategy=bert_avg_embeddings)

def create_bert_max_embedder(bertTokenizer, bertModel, filepath):
    return BertEmbedder(bertTokenizer, bertModel, device=device, embedding_strategy=bert_max_embeddings)

def create_bert_sif_embedder(bertTokenizer, bertModel, filepath):
    unigram = get_unigrams(filepath, bertTokenizer)
    return BertEmbedder(bertTokenizer, bertModel, device=device,
                                embedding_strategy=BertWeightedEmbeddings(unigram, device=device))

EMBEDER_NAME_TO_FACTORY = {'bert_cls': create_bert_cls_embedder, 'bert_max': create_bert_max_embedder,
                           'bert_avg':create_bert_avg_embedder,'bert_sif': create_bert_sif_embedder}

def train_model(model, dataloader, start_lr, end_lr, num_epochs, with_decoder=False):
    losses = {}
    if not with_decoder:
        model.decoder = None
        losses = {'clusterer_loss': constant_loss_weight_fn(1.0)}
    else:
        decoder_weight = 0.5
        losses = {'clusterer_loss': constant_loss_weight_fn(1 - decoder_weight),
                  'decoder_loss': constant_loss_weight_fn(decoder_weight)}

    gamma = (end_lr / start_lr) ** (1 / num_epochs)
    train(model, dataloader, losses, start_lr, num_epochs=num_epochs, gamma=gamma)


def save_metrics(model, dataloader, filename):
    with open(filename, 'w') as metric_file:
        metric_file.write(str(calculate_metrics(model, dataloader)) + '\n')
        metric_file.write(str(classify(model, dataloader)))

def _embed_data(parser):
    bertTokenizer = create_tokenizer()
    bertModel = create_model().to(device)
    for datasetname in parser.datasets.split(','):
        for embedtype in parser.embed_types.split(','):
            print('Embedding', datasetname, 'with', embedtype)
            input_path = os.path.join(parser.data_dir, '{}.txt'.format(datasetname))
            output_path = os.path.join(parser.embeds_dir, '{}_{}.npy'.format(datasetname, embedtype))
            embedder_factory = EMBEDER_NAME_TO_FACTORY[embedtype]
            embedder = embedder_factory(bertTokenizer, bertModel, input_path)

            read_n_save(input_path, output_path, embedder)

def _train_autoencoders(parser):
    for dataset_name in parser.datasets.split(','):
        for emb_name in parser.embed_types.split(','):
            for i in range(parser.num):
                print('Training ae on ', dataset_name, emb_name, '#', i)
                emb_filename = os.path.join(parser.embeds_dir, '{}_{}.npy'.format(dataset_name, emb_name))
                labels_filename = os.path.join(parser.embeds_dir, '{}_label.npy'.format(dataset_name))
                dataloader = read_saved(emb_filename, labels_filename, device=device)

                input_size = dataloader.dataset[0]['input'].shape[0]
                encoder = createEncoder(input_size).to(device)
                decoder = createDecoder(input_size).to(device)
                losses = {'decoder_loss': constant_loss_weight_fn(1.0)}

                gamma = (parser.end_lr / parser.start_lr) ** (1 / parser.num_epochs)
                train_autoencoder(encoder, decoder, dataloader, losses, parser.start_lr, gamma=gamma, num_epochs=parser.num_epochs)

                cluster_centers = init_cluster_centers(encoder, dataloader, DATEST_TO_NUMCLUSTERS[dataset_name])
                clusterer = XieClusterer(torch.tensor(cluster_centers, requires_grad=True, device=device),
                                         trainable_centers=True)
                model = DeepClusteringModel(encoder, clusterer, decoder=decoder)

                output_filename = os.path.join(parser.ae_dir, '{}_{}_ae_{}.pt'.format(dataset_name, emb_name, i))
                torch.save(model, output_filename)

def _train_models(parser):
    for data_name in parser.datasets.split(','):
        for emb_name in parser.embed_types.split(','):
            for num in range(parser.num):
                emb_filename = os.path.join(parser.data_dir, '{}_{}.npy'.format(data_name, emb_name))
                labels_filename = os.path.join(parser.data_dir, '{}_label.npy'.format(data_name))
                ae_filename = os.path.join(parser.ae_dir, '{}_{}_ae_{}.pt'.format(data_name, emb_name, num))

                dataloader = read_saved(emb_filename, labels_filename, device=device)

                model = torch.load(ae_filename)
                model.decoder = None

                metrics_filename = os.path.join(parser.metrics_dir,
                                                'metrics_{}_{}_{}_initial.json'.format(data_name, emb_name, num))
                save_metrics(model, dataloader, metrics_filename)
                for loss_name in parser.losses.split(','):
                    print('Training on ', data_name, emb_name, '#', num, 'with loss', loss_name)

                    loss_fn = LOSSES[loss_name]

                    model = torch.load(ae_filename)
                    model.clusterer.loss_fn = loss_fn
                    train_model(model, dataloader, parser.start_lr, parser.end_lr, parser.num_epochs, with_decoder=False)
                    model_filename = os.path.join(parser.models_dir, 'model_{}_{}_{}.pt'.format(data_name, emb_name, num))
                    metrics_filename = os.path.join(parser.metrics_dir, 'metrics_{}_{}_{}.json'.format(data_name, emb_name, num))
                    torch.save(model, model_filename)
                    save_metrics(model, dataloader, metrics_filename)

                    model = torch.load(ae_filename)
                    model.clusterer.loss_fn = loss_fn
                    train_model(model, dataloader, parser.start_lr, parser.end_lr, parser.num_epochs, with_decoder=True)
                    model_filename = os.path.join(parser.models_dir, 'model_{}_{}_{}_dec.pt'.format(data_name, emb_name, num))
                    metrics_filename = os.path.join(parser.metrics_dir, 'metrics_{}_{}_{}_dec.json'.format(data_name, emb_name, num))
                    torch.save(model, model_filename)
                    save_metrics(model, dataloader, metrics_filename)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['embed', 'train_ae', 'train_model'])
    parser.add_argument('--datasets', default='SearchSnippets,Biomedical,StackOverflow')
    parser.add_argument('--embed_types', default='bert_cls,bert_avg,bert_sif,bert_max')
    parser.add_argument('--losses', default='kl_div,cross_entropy,bce,dot_product')
    parser.add_argument('--num', type=int, default=5, help='Number of autoencoders trained')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--embeds_dir', default='output/embeds')
    parser.add_argument('--ae_dir', default='output/ae')
    parser.add_argument('--models_dir', default='output/models')
    parser.add_argument('--metrics_dir', default='output/metrics')
    parser.add_argument('--num_epochs', default=800, type=int)
    parser.add_argument('--start_lr', default=3e-3, type=float)
    parser.add_argument('--end_lr', default=1e-4, type=float)

    global device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    args=parser.parse_args()
    if args.action == 'embed':
        print('Embedding data')
        _embed_data(args)
    if args.action == 'train_ae':
        _train_autoencoders(args)
    if args.action == 'train_model':
        _train_models(args)


