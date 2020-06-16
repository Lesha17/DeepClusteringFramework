import numpy as np
import torch
import tqdm
import os
from torch.utils.data import DataLoader
from typing import List, Dict

class ResultComputer:
    def compute(self, dependencies: Dict[str, object]):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

class ComputingStep:
    def __init__(self, result_computer: ResultComputer, dependencies: dict[str, 'ComputingStep']):
        super(self, ComputingStep).__init__()
        self.dependencies = dependencies
        self.result_computer = result_computer

    def get_result(self) -> object:
        if not self.has_result():
            dependencies = {k: v.get_result() for k, v in self.dependencies.items()}
            self.save_result(self.result_computer.compute(dependencies))

        assert self.has_result()
        return self.load_result()

    def has_result(self) -> bool:
        raise NotImplementedError()

    def load_result(self) -> object:
        raise NotImplementedError()

    def save_result(self, result : object):
        raise NotImplementedError()

class RuntimeComputingStep(ComputingStep):
    def __init__(self, result_computer: ResultComputer, dependencies: Dict[str, ComputingStep]):
        super(self, RuntimeComputingStep).__init__(result_computer, dependencies)
        self.result = None

    def has_result(self) -> bool:
        return self.result is not None

    def load_result(self) -> object:
        return self.result

    def save_result(self, result : object):
        self.result = result

class FileComputingStep(ComputingStep):
    def __init__(self, result_computer: ResultComputer, dependencies: Dict[str, ComputingStep], filepath: str):
        super(self, FileComputingStep).__init__(result_computer, dependencies)
        self.result_filepath = filepath

    def has_result(self) -> bool:
        return os.path.exists(self.result_filepath)

    def load_result(self) -> object:
        return torch.load(self.result_filepath)

    def save_result(self, result : object):
        torch.save(result, self.result_filepath)

def make_step(result_computer: ResultComputer, dependencies: Dict[str, ComputingStep], filepath = None):
    if filepath is not None:
        return FileComputingStep(result_computer, dependencies, filepath)
    else:
        return RuntimeComputingStep(result_computer, dependencies)


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