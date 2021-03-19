import torch
from typing import Union, List, Tuple, Dict
from clustering_tool.datapipeline import ResultComputer
from clustering_tool.embedders.bert import bert_cls_embeddings
from transformers import BertTokenizer, BertModel, BertForMaskedLM

class Tokenizer:
    def __call__(self, *inputs, **kwargs) -> Union[Tuple, Dict[str, object]]:
        return self.tokenize(*inputs, **kwargs)

    def tokenize(self, text, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

class MyBertTokenizer:
    def __init__(self, bert_tokenizer: BertTokenizer, max_len: int, target_device):
        super(MyBertTokenizer, self).__init__()

        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.device = target_device

    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, return_attention_mask=True,
                                            return_tensors='pt', max_length=self.max_len, pad_to_max_length=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return tokens


class TokenizerOutputsComputer(ResultComputer):
    def __init__(self, tokenizer: Tokenizer, text_key: str):
        '''

        :param tokenizer:
        :param text_key: key of text in dataset if dataset with dictionary samples
        '''
        super(TokenizerOutputsComputer, self).__init__()

        self.tokenizer = tokenizer
        self.text_key = text_key

    def compute(self, dependencies: Dict[str, object]):
        dataset = dependencies['dataset']

        result = []
        for sample in dataset:
            if isinstance(sample, dict):
                text = dataset[self.text_key]
            elif isinstance(sample, list):
                text = dataset[0]
            else:
                text = sample
            tokenizer_outputs = self.tokenizer(text)
            result.append(tokenizer_outputs)

        return result


class Embedder:
    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)

    def embed(self, tokenizer_output: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]):
        raise NotImplementedError

class MyBertEmbedder(Embedder):
    def __init__(self, bert_model: BertModel):
        super(MyBertEmbedder, self).__init__()

        self.bert_model = bert_model

    def embed(self, tokenizer_output: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]):
        if isinstance(tokenizer_output, dict):
            embedded_text = self.embedder(**tokenizer_output)
        elif isinstance(tokenizer_output, list):
            embedded_text = self.embedder(*tokenizer_output)
        else:
            embedded_text = self.embedder(tokenizer_output)

        embedded_text = embedded_text.squeeze(0) # zero dimension is 1

        return embedded_text


class TextEmbeddingsComputer(ResultComputer):
    def __init__(self, embedder: Embedder):
        super(TextEmbeddingsComputer, self).__init__()
        self.embedder = embedder

    def compute(self, dependencies: Dict[str, object]):
        tokenizer_outputs_list = dependencies['tokenizer_output']
        result = []
        for tokenizer_outputs in tokenizer_outputs_list:
            result.append(self.embedder(tokenizer_outputs))

        return result


