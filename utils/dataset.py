import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class Dataset(Dataset):
    def __init__(self, root_dir, flag='train', transform=None):
        super().__init__()
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.transform = transform
        self.image_dir = os.path.join(root_dir, flag, "images")
        self.image_paths = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir) if x.endswith(".png", ".jpg", ".jpeg")]
        self.mask_dir = os.path.join(root_dir, flag, "masks")
        self.mask_paths = [os.path.join(self.mask_dir, x) for x in os.listdir(self.mask_dir) if x.endswith(".png", ".jpg", ".jpeg")]
        assert len(self.image_paths) == len(self.mask_paths), f"Number of images and masks do not match in {self.flag} dataset"
        assert len(self.image_paths) > 0, f"No images found in {self.flag} dataset"


    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image
    
    def __len__(self):
        return len(self.image_paths), len(self.mask_paths)

    
      