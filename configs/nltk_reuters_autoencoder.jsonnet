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
        "type": "auto_encoder",
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
        "decoder": {
            "type": "auto_regressive_seq_decoder",
            "decoder_net": {
                "type": "lstm_cell",
                "decoding_dim": 10,
                "target_embedding_dim": embedding_dim
            },
            "max_decoding_steps": 1000,
            "target_embedder": {
                "embedding_dim": embedding_dim,
                "pretrained_file": "data/model.vec",
                 "trainable": false
            }
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