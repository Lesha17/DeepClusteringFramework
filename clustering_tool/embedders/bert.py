import torch
import tqdm
from transformers import BertModel, BertTokenizer
from collections import Counter

def create_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def create_model():
    return BertModel.from_pretrained('bert-base-uncased')

def bert_pooler_embeddings(hidden, pooler, inputs):
    return pooler

def bert_avg_embeddings(hidden, pooler, inputs):
    return torch.mean(hidden, dim=1)

def bert_max_embeddings(hidden, pooler, inputs):
    return torch.max(hidden, dim=1).values

def bert_cls_embeddings(hidden, pooler, inputs):
    return hidden[:, 0]

def BertWeightedEmbeddings(unigram, device = 'cuda'):
    def get_unigram_tensor(ids):
        result=torch.zeros(ids.shape, device=device)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                id = ids[i, j]
                if id in unigram:
                    result[i, j] = unigram[id]
        return result

    def weighted_embeddings(hidden, pooler, inputs):
        weight = (0.1 / (0.1 + get_unigram_tensor(inputs['input_ids']))) * inputs['attention_mask'] \
                 / torch.sum(inputs['attention_mask'], dim=1)
        return torch.sum(hidden * weight.unsqueeze(-1), dim=1)
    return weighted_embeddings

class BertEmbedder(torch.nn.Module):
    def __init__(self, tokenizer, bertModel, device = 'cuda', max_len = 80, embedding_strategy=bert_cls_embeddings):
        super(BertEmbedder, self).__init__()
        self.tokenizer = tokenizer
        self.bertModel = bertModel
        self.device = device
        self.max_len = max_len
        self.embedding_strategy = embedding_strategy

    def forward(self, text):
        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=self.max_len,
                                           pad_to_max_length=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeds = self.embedding_strategy(*self.bertModel(**inputs), inputs)
        return embeds
