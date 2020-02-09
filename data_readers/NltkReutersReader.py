from typing import Dict

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import TextField, ArrayField, MultiLabelField
from nltk.corpus import reuters

@DatasetReader.register("nltk_reuters_reader")
class NltkReutersReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]):
        super(NltkReutersReader, self).__init__(lazy=True)
        self.token_indexers = token_indexers

    def _read(self, file_path: str):
        files = [f for f in reuters.fileids() if f.startswith(file_path)]
        for file in files:
            yield self.text_to_instance([Token(word) for word in reuters.words(file)], reuters.categories(file))


    def text_to_instance(self, words, categories):
        fields = { "sentence": TextField(words, self.token_indexers), "label": MultiLabelField(categories) }
        return Instance(fields)

