local vocab_size = 47236;
local labels_count = 103;

{
    "train_data_path": "train",
    "dataset_reader": {
        "type": "reuters_reader",
    },
    "model": {
        "type": "deep_clustering",
        "embedder": {
            "type": "pass_through",
            "hidden_dim": vocab_size
        },
        "encoder": {
            "type": "feedforward",
            "input_dim": vocab_size,
            "num_layers": 3,
            "hidden_dims": 10,
            "activations": "relu"
        },
        "clusterer": {
            "type": "xie_clusterer",
            "num_clusters": labels_count,
            "embedding_size": 10
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