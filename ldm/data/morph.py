import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Morph(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 gene_path=None,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        #self.image_paths=os.listdir(data_root)
        self._length = len(self.image_paths)
        if gene_path is not None:
            gene_vector=np.load(gene_path)
            self.labels = {
                "relative_file_path_": [l for l in self.image_paths],
                "file_path_": [os.path.join(self.data_root, l)
                            for l in self.image_paths],
                #"file_path_": [l for l in self.image_paths],
                "gene_vector": gene_vector
            }
        else:
            self.labels = {
                "relative_file_path_": [l for l in self.image_paths],
                "file_path_": [os.path.join(self.data_root, l)
                            for l in self.image_paths],
            }

        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        #print(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

class Train(Morph):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/train_img_order.txt", 
                         data_root="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/train", **kwargs)
class Test(Morph):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/test_img_order.txt", 
                         data_root="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/test", flip_p=flip_p, **kwargs)
        
#class Merfish(Morph):
#    def __init__(self, flip_p=0., **kwargs):
#        super().__init__(txt_file="/data/project/wangning/latent-diffusion/ldm/data/merfish.txt", 
#                         data_root="/data/project/wangning/latent-diffusion/ldm/data/merfish_clean/train", flip_p=flip_p, **kwargs)