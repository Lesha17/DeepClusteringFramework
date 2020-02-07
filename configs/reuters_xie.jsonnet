local vocab_size = 47236;
local pca_dim = 2000;
local labels_count = 103;

{
    "train_data_path": "train",
    "validation_data_path": "train", // there is no "validation" dataset in sklearn rcv1
    "dataset_reader": {
        "type": "reuters_reader",
    },
    "model": {
        "num_clusters": labels_count,
        "type": "deep_clustering",
        "embedder": {
            "type": "pca",
            "weights_file": "data/reuters_pca.joblib"
        },
        "encoder": {
            "type": "feedforward",
            "input_dim": pca_dim,
            "num_layers": 2,
            "hidden_dims": [100, 20],
            "activations": "relu"
        },
        "clusterer": {
            "type": "xie_clusterer",
            "num_clusters": labels_count,
            "embedding_size": 20
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
            "lr": 0.0001
        },
        "num_serialized_models_to_keep": -1,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "patience": 2
        },
        "validation_metric": "-loss"
    }
}