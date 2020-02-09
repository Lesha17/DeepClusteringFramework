import torch

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_decoders import SeqDecoder
from allennlp.nn import Initializer
from allennlp.nn.util import get_text_field_mask

@Model.register("auto_encoder")
class AutoEncoder(Model):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 decoder: SeqDecoder,
                 embedders: TextFieldEmbedder = None):
        super(AutoEncoder, self).__init__(vocab)

        self.encoder = encoder
        self.decoder = decoder
        self.embedders = embedders

    def forward(self, sentence, label=None):
        mask = get_text_field_mask(sentence)
        Initializer
        if self.embedders:
            x = self.embedders(sentence)
        else:
            embedded_representations = []
            for key, value in sentence.items():
                embedded_representations.append(value)
            x = torch.cat(embedded_representations, dim=-1)

        h = self.encoder(x, mask)
        source_mask = get_text_field_mask({'h': h.unsqueeze(1)})
        encoder_out = {"encoder_outputs": h.unsqueeze(1), "source_mask": source_mask}
        sentence["tokens"] = sentence["single_id"]
        output_dict = self.decoder(encoder_out, sentence)

        return output_dict

