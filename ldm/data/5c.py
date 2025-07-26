import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class morph_5c(Dataset):
    def __init__(self, npy_path, gene_path=None, size=None, interpolation=Image.BILINEAR, flip=False):
        self.data = np.load(npy_path)
        self.size = size
        self.interpolation = interpolation
        self.flip = flip
        self._length = self.data.shape[0]
        if gene_path is not None:
            gene_vector=np.load(gene_path)
            self.labels = {   
                "gene_vector": gene_vector
            }
        else:
            self.labels=None

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.labels:
            example = dict((k, self.labels[k][i]) for k in self.labels)
        else:
            example = {}
        image = self.data[i]
        example['image']=(image / 127.5 - 1.0).astype(np.float32)
        return example

class Train(morph_5c):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Val(morph_5c):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)