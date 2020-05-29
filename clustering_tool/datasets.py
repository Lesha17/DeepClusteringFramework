import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

def read_n_save(input_file, out_file, embedder):
    data = []

    with open(input_file, 'r') as file:
        for text in tqdm.tqdm(file, disable=True):
            emb = embedder(text)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().detach().numpy()
            data.append(emb.squeeze(0))

    np.save(out_file, np.array(data))
    print('Read and save completed successfully')

def read_saved(x_file, label_file, device='cuda'):
    x = np.load(x_file)
    labels = np.load(label_file)

    ind = list(range(x.shape[0]))
    np.random.shuffle(ind)
    x = x[ind]
    labels = labels[ind]

    data = [{'input': torch.as_tensor(input, device=device, dtype=torch.float), 'label': torch.as_tensor(label, device=device)} for input, label in list(zip(x, labels))]
    return DataLoader(data, batch_size=256, shuffle=True)