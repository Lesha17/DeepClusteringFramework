import logging

from allennlp.data import DatasetReader, Instance
from sklearn.datasets import fetch_rcv1

from data_readers.fields.VectorField import VectorField

logger = logging.getLogger(__name__)

@DatasetReader.register("reuters_reader")
class ReutersReader(DatasetReader):
    def __init__(self):
        super(ReutersReader, self).__init__(lazy=True)

    def _read(self, file_path: str):
        # AllenNLP architecure is built for reading from file, however, it's easier to obtain reuters from sklearn
        # So, since there's no way to provide data another way but DatasetReader, just obtain it from sklearn
        #     and use file_path argument only to differ train, test and validation data

        dataset = fetch_rcv1(data_home='data', subset=file_path, shuffle=True)
        for (data, target) in zip(dataset.data, dataset.target):
            yield self.text_to_instance(data, target[0])

    def text_to_instance(self, data, target):
        text_tfidf = VectorField(data.toarray(), namespace="tfidf")
        fields = {"sentence": text_tfidf}

        if target is not None:
            label_field = VectorField(target.toarray())
            fields["label"] = label_field

        return Instance(fields)
