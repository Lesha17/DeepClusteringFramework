local vocab_size = 47236;
local embedding_dim = 768;
local labels_count = 103;
local bert_model = "bert-base-uncased";

{
    "train_data_path": "train",
    "validation_data_path": "train",
    "dataset_reader": {
        "type": "nltk_reuters_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "pretrained_model": bert_model
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        },
        "max_len": 80
    },
    "model": {
        "type": "deep_clustering",
        "num_clusters": labels_count,
        "embedders": {
            "type": "basic",
            "token_embedders":  {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "requires_grad": true
                }
            },
            "embedder_to_indexer_map": {"bert": ["bert"]},
            "allow_unmatched_keys": true
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "bert-base-uncased"
        },
        "clusterer": {
            "type": "xie_clusterer",
            "num_clusters": labels_count,
            "embedding_size": embedding_dim
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": 50,
        "patience": 10,
        "optimizer": {
            "type": "adam",
            "lr": 1e-5
        },
        "num_serialized_models_to_keep": 5,
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.9
        },
        "validation_metric": "-loss"
    }
}