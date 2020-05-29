import torch
from collections import Counter
import tqdm

def get_unigrams(file, tokenizer):
    word_count = Counter()
    total_cnt = 0
    with open(file, 'r') as snippets_file:
        for snippet in tqdm.tqdm(snippets_file, disable=True):
            with torch.no_grad():
                ids = tokenizer.encode(snippet, add_special_tokens=True)
                for id in ids:
                    word_count[id]+= 1
                    total_cnt += 1

    unigram = {}
    for word, cnt in word_count.items():
        unigram[word] = cnt / total_cnt

    return unigram