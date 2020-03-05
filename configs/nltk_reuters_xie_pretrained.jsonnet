local vocab_size = 47236;
local embedding_dim = 300;
local labels_count = 103;

{
    "train_data_path": "train",
    "validation_data_path": "train", // there is no "validation" dataset in sklearn rcv1
    "dataset_reader": {
        "type": "nltk_reuters_reader",
        "token_indexers": {
            "single_id": "single_id"
        }
    },
    "model": {
        "type": "deep_clustering",
        "num_clusters": labels_count,
        "embedders": {
            "single_id": {
                "type": "embedding",
                "embedding_dim": embedding_dim,
                "pretrained_file": "data/model.vec",
                 "trainable": false
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": 10,
            "bidirectional": false
        },
        "encoder_initializer": {
            "type": "pretrained",
            "weights_file_path": "models/nltk_reuters_ae/best.th"
        },
        "clusterer": {
            "type": "xie_clusterer",
            "num_clusters": labels_count,
            "embedding_size": 10
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 10
    },
    "trainer": {
        "num_epochs": 50,
        "patience": 10,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "num_serialized_models_to_keep": 5,
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.9
        },
        "validation_metric": "-loss"
    }
}