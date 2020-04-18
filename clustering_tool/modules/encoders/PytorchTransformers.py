import torch

from allennlp.common import FromParams
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler
from  allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import PretrainedBertEmbedder

class BertEmbeddingStrategy:
    def __init__(self):
        pass

    def __call__(self, hidden, pooler):
        return self.get_embeddings(hidden, pooler)

    def get_embeddings(self, hidden, pooler):
        raise NotImplementedError()

class BertPoolerEmbedding(BertEmbeddingStrategy):
    def get_embeddings(self, hidden, pooler):
        return pooler

class BertAvgEmbedding(BertEmbeddingStrategy):
    def get_embeddings(self, hidden, pooler):
        return torch.mean(hidden, dim=1)

class BertMaxEmbedding(BertEmbeddingStrategy):
    def get_embeddings(self, hidden, pooler):
        return torch.max(hidden, dim=1).values

class BertClsEmbedding(BertEmbeddingStrategy):
    def get_embeddings(self, hidden, pooler):
        return hidden[:, 0]

@Seq2VecEncoder.register('pytorch-transformers')
class PytorchTransformersEncoder(Seq2VecEncoder, FromParams):
    def __init__(self,
                 bertModel,
                 embedding_strategy: BertEmbeddingStrategy):
        super(PytorchTransformersEncoder, self).__init__()
        self.bertModel = bertModel
        self.embedding_strategy = embedding_strategy

    def forward(self, ids, mask):
        return self.embedding_strategy(*self.bertModel(ids, mask))



