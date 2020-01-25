{
    "train_data_path": "", # TODO find dataset
    "validation_data_path": "", # TODO
    "dataset_reader": {
        "type": "text-corpus-reader", # TODO read data
        "token_indexers": {
            "single_id":"single_id"
        }
    },
    "model": {
        "type": "deep_clustering",
        "embedder": {
            "token_embedders": { # TODO is it correct in this case
                "single_id": {
                    "type": "embedding",
                    "embedding_dim": 50,
                    "pretrained_file": "model.vec",
                    "trainable": false
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 100,

            "bidirectional": true
        },
        "clusterer": {
            "type": "xie_clusterer",
            "num_clusters": 10,
            "embedding_size": 100
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 100
    },
    "trainer": {
        "num_epochs": 50,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 10
    }
}