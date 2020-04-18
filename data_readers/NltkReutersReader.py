from typing import Dict

from allennlp.data import DatasetReader, Instance, TokenIndexer
from allennlp.data.fields import TextField, MultiLabelField
from allennlp.data.tokenizers.tokenizer import Tokenizer
import nltk
from nltk.corpus import reuters
from nltk.corpus.reader.util import concat


def _read_sents(stream):
    sents = []
    for para in reuters._para_block_reader(stream):
        sents.extend([" ".join(sent.split()) for sent in reuters._sent_tokenizer.tokenize(para)])
    return sents


def sents(fileids=None):
    """
    :return: the given file(s) as a list of raw sentences or utterances.
    :rtype: list(str)
    """
    return concat(
        [
            reuters.CorpusView(path, reuters._read_sent_block, encoding=enc)
            for (path, enc, fileid) in reuters.abspaths(fileids, True, True)
        ]
    )

@DatasetReader.register("nltk_reuters_reader")
class NltkReutersReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], max_len=-1):
        super(NltkReutersReader, self).__init__(lazy=True)

        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_len = max_len

    def _read(self, file_path: str):
        nltk.download('reuters')
        nltk.download('punkt')

        files = [f for f in reuters.fileids() if f.startswith(file_path)]
        for file in files:
            categories = reuters.categories(file)
            for sent in sents(file):
                tokens = self.tokenizer.tokenize(sent)
                # allennlp does not provide correct way to specify max sequence length, so it should be done in datareader
                if self.max_len > 0 and len(tokens) > self.max_len:
                    tokens = tokens[:self.max_len]
                yield self.text_to_instance(tokens, categories)

    def text_to_instance(self, words, categories):
        fields = { "sentence": TextField(words, self.token_indexers), "label": MultiLabelField(categories) }
        return Instance(fields)

