from typing import Dict, List, Union
import torch

class EmbedsCombiner:
    def __call__(self, *args, **kwargs):
        return self.combine(*args, **kwargs)

    def combine(self, tokens: Union[torch.Tensor, List[torch.Tensor]],
                mask: Union[torch.Tensor, List[torch.Tensor]],
                embeds: Union[torch.Tensor, List[torch.Tensor]]):
        raise NotImplementedError


class CLSEmbedsCombiner(EmbedsCombiner):
    '''
    Returns the first ('classification') token embedding
    '''

    def combine(self, tokens: Union[torch.Tensor, List[torch.Tensor]],
                mask: Union[torch.Tensor, List[torch.Tensor]],
                embeds: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(embeds, torch.Tensor):
            return embeds[:, 0]
        else:
            return [text_embeds[0] for text_embeds in embeds]



